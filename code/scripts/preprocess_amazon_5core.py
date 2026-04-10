from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def parse_gz(path: Path):
    with gzip.open(path, "r") as handle:
        for line in handle:
            yield eval(line.replace(b"true", b"True").replace(b"false", b"False"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Amazon 5-core reviews into sequential TXT format.")
    parser.add_argument("--dataset", required=True, help="Example: Toys_and_Games_5")
    parser.add_argument("--input", default=None, help="Optional path to reviews_*.json.gz")
    parser.add_argument("--output", default=None, help="Optional output TXT path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input) if args.input else root / "data" / "raw" / f"reviews_{args.dataset}.json.gz"
    output_path = Path(args.output) if args.output else root / "data" / f"{args.dataset}_time.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for review in parse_gz(input_path):
        rows.append((review["reviewerID"], review["asin"], int(review["unixReviewTime"])))

    user_map = {}
    item_map = {}
    user_hist = {}
    for user, item, timestamp in rows:
        if user not in user_map:
            user_map[user] = len(user_map) + 1
            user_hist[user_map[user]] = []
        if item not in item_map:
            item_map[item] = len(item_map) + 1
        user_hist[user_map[user]].append((item_map[item], timestamp))

    with output_path.open("w") as handle:
        for user_id, seq in user_hist.items():
            for item_id, timestamp in sorted(seq, key=lambda pair: pair[1]):
                handle.write(f"{user_id} {item_id} {timestamp}\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
