import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch_geometric.nn import SAGEConv

from fetch_elliptic import verify

LOCAL_COLUMNS = 93
LAST_STEP = 49


@dataclass
class Transactions:
    features: np.ndarray
    steps: np.ndarray
    labels: np.ndarray
    directed_edges: np.ndarray


def load_elliptic(folder):
    verify(folder)
    path = folder / "elliptic_txs_features.csv"
    with path.open(encoding="utf-8") as stream:
        if len(stream.readline().strip().split(",")) != 167:
            raise ValueError("Unexpected Elliptic feature layout")
    dtypes = {0: np.int64, 1: np.int16}
    dtypes.update({index: np.float32 for index in range(2, LOCAL_COLUMNS + 2)})
    table = pd.read_csv(path, header=None, usecols=list(dtypes), dtype=dtypes)
    ids = pd.Index(table[0])
    if not ids.is_unique:
        raise ValueError("Transaction IDs are not unique")
    classes = pd.read_csv(folder / "elliptic_txs_classes.csv").set_index("txId")
    if not classes.index.is_unique or len(classes) != len(ids):
        raise ValueError("Class file must define one row per transaction")
    labels = classes["class"].map({"unknown": -1, "1": 1, "2": 0}).reindex(ids)
    if labels.isna().any():
        raise ValueError("Missing or invalid transaction label")
    links = pd.read_csv(folder / "elliptic_txs_edgelist.csv")
    sources = ids.get_indexer(links["txId1"])
    targets = ids.get_indexer(links["txId2"])
    if (sources < 0).any() or (targets < 0).any():
        raise ValueError("Graph includes an unknown endpoint")
    steps = table[1].to_numpy()
    features = table.iloc[:, 2:].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all() or set(steps) != set(range(1, LAST_STEP + 1)):
        raise ValueError("Invalid features or incomplete time steps")
    return Transactions(features, steps, labels.to_numpy(dtype=np.int8), np.stack([sources, targets]))


def causal_edges(dataset, last_step):
    sources, targets = dataset.directed_edges
    source_steps, target_steps = dataset.steps[sources], dataset.steps[targets]
    usable = (source_steps <= target_steps) & (target_steps <= last_step)
    return dataset.directed_edges[:, usable]


def graph_prefix(dataset, normalized, cutoff):
    included = np.flatnonzero(dataset.steps <= cutoff)
    to_local = np.full(len(dataset.steps), -1, dtype=np.int64)
    to_local[included] = np.arange(len(included))
    edges = torch.from_numpy(to_local[causal_edges(dataset, cutoff)])
    values = torch.from_numpy(normalized[included])
    return included, values, edges


class RiskSAGE(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.sage = SAGEConv(feature_count, 32, aggr="mean")
        self.output = nn.Linear(32, 1)

    def forward(self, values, edges):
        hidden = self.sage(values, edges).relu()
        return self.output(hidden).squeeze(-1)


def graph_scores(dataset, normalized, config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = RiskSAGE(normalized.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train_idx, train_x, train_edges = graph_prefix(dataset, normalized, config["train_last"])
    labels = torch.as_tensor(dataset.labels[train_idx], dtype=torch.float32)
    known = labels >= 0
    positives = (labels == 1).sum()
    weight = (labels == 0).sum() / positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    model.train()
    for _ in range(config["graph_epochs"]):
        optimizer.zero_grad()
        scores = model(train_x, train_edges)
        loss = loss_fn(scores[known], labels[known])
        loss.backward()
        optimizer.step()

    model.eval()
    # Incoming edges never start after their target. Earlier scores therefore
    # remain independent of later nodes in the one-hop full-graph pass.
    full_idx, full_x, full_edges = graph_prefix(dataset, normalized, LAST_STEP)
    with torch.inference_mode():
        full_scores = model(full_x, full_edges).numpy()
        early_scores = model(train_x, train_edges).numpy()
    if not np.allclose(early_scores, full_scores[train_idx], atol=2e-5, rtol=2e-5):
        raise RuntimeError("Graph scores changed after appending future nodes")
    result = np.empty(len(dataset.steps), dtype=np.float32)
    result[full_idx] = full_scores
    return result


def rate(numerator, denominator):
    return round(float(numerator / denominator), 6) if denominator else None


def label_summary(labels):
    return {
        "total": int(len(labels)),
        "illicit": int((labels == 1).sum()),
        "licit": int((labels == 0).sum()),
        "unknown": int((labels < 0).sum()),
        "annotation_fraction": rate((labels >= 0).sum(), len(labels)),
    }


def review(labels, scores, steps, budget):
    selected = []
    for step in np.unique(steps):
        rows = np.flatnonzero(steps == step)
        order = np.argsort(-scores[rows], kind="stable")
        selected.append(rows[order[:budget]])
    picked = labels[np.concatenate(selected)]
    summary = label_summary(picked)
    positives, negatives, unknown = summary["illicit"], summary["licit"], summary["unknown"]
    return {
        **summary,
        "confirmed_precision_lower": rate(positives, len(picked)),
        "potential_precision_upper": rate(positives + unknown, len(picked)),
        "annotated_precision": rate(positives, positives + negatives),
        "confirmed_recall": rate(positives, (labels == 1).sum()),
    }


def choose_budget(labels, scores, steps, candidates, minimum_precision):
    options = {budget: review(labels, scores, steps, budget) for budget in candidates}
    eligible = [budget for budget, result in options.items()
                if result["confirmed_precision_lower"] >= minimum_precision]
    if eligible:
        budget = max(eligible, key=lambda item: (options[item]["illicit"], -item))
    else:
        budget = max(options, key=lambda item: (options[item]["confirmed_precision_lower"], -item))
    return budget, options


def scored_metrics(labels, raw, probabilities):
    known = labels >= 0
    positives = (labels == 1).sum()
    return {
        "average_precision": round(float(average_precision_score(labels[known], raw[known])), 6)
        if positives and (labels == 0).any() else None,
        "brier": round(float(brier_score_loss(labels[known], probabilities[known])), 6)
        if known.any() else None,
    }


def evaluate(labels, raw, probabilities, steps, budget):
    return {
        **label_summary(labels),
        **scored_metrics(labels, raw, probabilities),
        "review": review(labels, raw, steps, budget),
    }


def run(dataset, config):
    train = dataset.steps <= config["train_last"]
    validation = (dataset.steps > config["train_last"]) & (dataset.steps <= config["validation_last"])
    test = dataset.steps > config["validation_last"]
    known_train = train & (dataset.labels >= 0)
    known_validation = validation & (dataset.labels >= 0)
    if min(len(np.unique(dataset.labels[mask])) for mask in (known_train, known_validation)) != 2:
        raise ValueError("Training and calibration require both known classes")

    labels = dataset.labels
    scaler = StandardScaler().fit(dataset.features[known_train])
    normalized = np.clip(scaler.transform(dataset.features), -12, 12).astype(np.float32)
    linear = LogisticRegression(class_weight="balanced", max_iter=500)
    linear.fit(normalized[known_train], labels[known_train])
    boost = HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=120, early_stopping=False, random_state=config["seed"]
    ).fit(dataset.features[known_train], labels[known_train])
    scores = {
        "logistic": linear.decision_function(normalized),
        "boosting": boost.decision_function(dataset.features),
        "graphsage": graph_scores(dataset, normalized, config),
    }
    budget, candidates = choose_budget(
        labels[validation], scores["logistic"][validation], dataset.steps[validation],
        config["budget_candidates"], config["minimum_validation_precision"],
    )
    sources, targets = dataset.directed_edges
    time_delta = dataset.steps[targets] - dataset.steps[sources]
    models = {}
    for name, raw in scores.items():
        calibrator = LogisticRegression().fit(raw[known_validation, None], labels[known_validation])
        probabilities = calibrator.predict_proba(raw[:, None])[:, 1]
        models[name] = {
            "platt_slope": round(float(calibrator.coef_[0, 0]), 6),
            "validation": evaluate(labels[validation], raw[validation], probabilities[validation],
                                   dataset.steps[validation], budget),
            "test": evaluate(labels[test], raw[test], probabilities[test], dataset.steps[test], budget),
            "test_steps": {},
        }
        for step in np.unique(dataset.steps[test]):
            period = dataset.steps == step
            models[name]["test_steps"][str(step)] = evaluate(
                labels[period], raw[period], probabilities[period], dataset.steps[period], budget
            )

    coefficients = linear.coef_[0]
    largest = np.argsort(-np.abs(coefficients), kind="stable")[:8]
    return {
        "config": config,
        "splits": {name: label_summary(labels[mask])
                   for name, mask in (("train", train), ("validation", validation), ("test", test))},
        "graphs": {
            "all": int(len(time_delta)),
            "excluded_future": int((time_delta < 0).sum()),
            "same_step": int((time_delta == 0).sum()),
            "strictly_earlier": int((time_delta > 0).sum()),
            "train_edges": int(causal_edges(dataset, config["train_last"]).shape[1]),
            "prefix_scores_invariant": True,
        },
        "selected_budget": budget,
        "validation_budget_candidates": candidates,
        "logistic_coefficients": [
            {"local_feature": int(idx + 1), "standardized_coefficient": round(float(coefficients[idx]), 6)}
            for idx in largest
        ],
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark chronological transaction triage")
    parser.add_argument("--data", type=Path, default=Path("data/raw/elliptic_bitcoin_dataset"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark/metrics.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 1 <= config["train_last"] < config["validation_last"] < LAST_STEP:
        parser.error("Require 1 <= train_last < validation_last < 49")
    if not config["budget_candidates"] or min(config["budget_candidates"]) < 1:
        parser.error("Review budgets must be positive")
    if config["graph_epochs"] < 1 or not 0 <= config["minimum_validation_precision"] <= 1:
        parser.error("Require graph_epochs >= 1 and minimum_validation_precision in [0, 1]")
    with threadpool_limits(limits=2):
        metrics = run(load_elliptic(args.data), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
