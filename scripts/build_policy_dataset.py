import argparse
import json
from pathlib import Path


SEED_ROWS = [
    {"state": {"intent": "RECOMMEND_PLACE_NEARBY", "slots": {"DISH": "phở", "LOCATION": None}, "missing_slots": ["LOCATION"], "turn_count": 1, "last_utterance": "Tìm quán phở"}, "action": "ASK_SLOT"},
    {"state": {"intent": "RECOMMEND_PLACE_NEARBY", "slots": {"DISH": "bún bò", "LOCATION": "gần đây"}, "missing_slots": [], "turn_count": 2, "last_utterance": "gần đây"}, "action": "CLARIFY"},
    {"state": {"intent": "RECOMMEND_PLACE_NEARBY", "slots": {"DISH": "bún bò", "LOCATION": "quận 1"}, "missing_slots": [], "turn_count": 2, "last_utterance": "quận 1"}, "action": "RECOMMEND"},
    {"state": {"intent": "SMALL_TALK", "slots": {}, "missing_slots": [], "turn_count": 1, "last_utterance": "xin chào"}, "action": "RESPOND"},
    {"state": {"intent": "OUT_OF_SCOPE", "slots": {}, "missing_slots": [], "turn_count": 1, "last_utterance": "viết code cho tôi"}, "action": "FALLBACK"},
]


def main():
    parser = argparse.ArgumentParser(description="Build seed policy_train.jsonl")
    parser.add_argument("--output", default="data/policy_train.jsonl")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in SEED_ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created seed dataset: {out}")


if __name__ == "__main__":
    main()