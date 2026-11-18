import unittest

import numpy as np

from benchmark import Transactions, causal_edges, choose_budget, graph_prefix, review, scored_metrics


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.sample = Transactions(
            np.array([[10], [20], [30], [40]], dtype=np.float32),
            np.array([2, 1, 2, 4]),
            np.array([0, -1, 1, -1]),
            np.array([[0, 1, 1, 0, 3], [1, 0, 3, 2, 2]]),
        )

    def test_causality_and_directed_orientation(self):
        np.testing.assert_array_equal(causal_edges(self.sample, 2), [[1, 0], [0, 2]])
        np.testing.assert_array_equal(causal_edges(self.sample, 4), [[1, 1, 0], [0, 3, 2]])

    def test_sparse_prefix_keeps_local_row_indices(self):
        idx, values, edges = graph_prefix(self.sample, self.sample.features, 1)
        self.assertEqual(idx.tolist(), [1])
        self.assertEqual(values.tolist(), [[20]])
        self.assertEqual(tuple(edges.shape), (2, 0))

    def test_unknowns_use_budget_and_bound_precision(self):
        result = review(np.array([-1, 1, 0, 1]), np.array([1., .9, .8, .1]),
                        np.array([1, 1, 1, 2]), 1)
        self.assertEqual(result["confirmed_precision_lower"], .5)
        self.assertEqual(result["potential_precision_upper"], 1.)
        self.assertEqual(result["annotated_precision"], 1.)
        self.assertEqual(result["confirmed_recall"], .5)

    def test_budget_uses_each_validation_step(self):
        budget, options = choose_budget(
            np.array([-1, 1, 0, 1]), np.array([1., .9, .8, .1]), np.array([1, 1, 1, 2]), [1, 2], .5
        )
        self.assertEqual(budget, 2)
        self.assertEqual(options[2]["total"], 3)

    def test_unknowns_excluded_from_brier_and_ap(self):
        result = scored_metrics(np.array([-1, 1, 0]), np.array([3., 2., 1.]), np.array([1., .8, .2]))
        self.assertEqual(result, {"average_precision": 1., "brier": .04})
        empty = scored_metrics(np.array([-1]), np.array([1.]), np.array([.5]))
        self.assertEqual(empty, {"average_precision": None, "brier": None})
