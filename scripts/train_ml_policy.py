import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def state_to_text(state: Dict) -> str:
    intent = state.get("intent", "NONE")
    missing = ",".join(sorted(state.get("missing_slots", [])))
    slots = state.get("slots", {}) or {}
    filled = ",".join(sorted([f"{k}:{v}" for k, v in slots.items() if v]))
    turns = state.get("turn_count", 0)
    last = state.get("last_utterance", "")
    return f"intent={intent} | missing={missing} | filled={filled} | turns={turns} | last={last}"


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train ML policy from JSONL")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_jsonl(Path(args.train_jsonl))
    if not data:
        raise ValueError("No training data found.")

    X = [state_to_text(x["state"]) for x in data]
    y = [x["action"] for x in data]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xv = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=1500)
    clf.fit(Xv, y)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": clf}, args.output)

    print(f"Saved: {args.output}")
    print(f"Train size: {len(data)}")
    print(f"Classes: {list(clf.classes_)}")
    print(f"Train acc: {clf.score(Xv, y):.4f}")


if __name__ == "__main__":
    main()