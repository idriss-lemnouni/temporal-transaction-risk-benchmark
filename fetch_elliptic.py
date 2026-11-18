import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("data") / "manifest.json"
SOURCE = json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify(folder):
    for name, expected in SOURCE["files"].items():
        path = folder / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Unexpected size for {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {name}")


def main():
    parser = argparse.ArgumentParser(description="Acquire and verify Elliptic v1 locally")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    folder = args.output / "elliptic_bitcoin_dataset"
    if not all((folder / name).is_file() for name in SOURCE["files"]):
        import kagglehub

        kagglehub.dataset_download(SOURCE["dataset"], output_dir=str(args.output.resolve()))
    verify(folder)
    print(folder)


if __name__ == "__main__":
    main()
