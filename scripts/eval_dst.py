from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.ml_policy import SklearnMLPolicy
from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF


def norm_text(x: Any) -> str:
    return str(x).strip().lower()


def slots_to_set(slots: Dict[str, Any]) -> set:
    out = set()
    for k, v in (slots or {}).items():
        if v is None:
            continue
        out.add((norm_text(k), norm_text(v)))
    return out


def build_orchestrator(args) -> DialogueOrchestrator:
    intent_model = IntentClassifierHF(args.intent_model_path)
    slot_model = SlotExtractorHF(args.slot_model_path)
    rule_policy = RuleBasedPolicy(rules_path=args.rules_path) if args.rules_path else RuleBasedPolicy()
    ml_policy = SklearnMLPolicy(args.ml_policy_path) if args.ml_policy_path else None

    llm_policy = None
    if args.use_llm:
        llm_policy = HuggingFaceLLMPolicy(
            model_name=args.llm_model,
            device=args.llm_device,
            torch_dtype=args.llm_dtype,
        )

    policy = HybridPolicy(
        rule_policy=rule_policy,
        ml_policy=ml_policy,
        llm_policy=llm_policy,
        ml_conf_threshold=args.ml_threshold,
        debug=True,
    )

    return DialogueOrchestrator(intent_model=intent_model, slot_model=slot_model, policy=policy)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DST quality from gold jsonl")
    parser.add_argument("--intent_model_path", required=True)
    parser.add_argument("--slot_model_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--rules_path", default=None)
    parser.add_argument("--ml_policy_path", default=None)
    parser.add_argument("--ml_threshold", type=float, default=0.7)

    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--llm_dtype", default="auto", choices=["auto", "float16", "float32"])
    args = parser.parse_args()

    orchestrator = build_orchestrator(args)

    rows = []
    for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))

    # group by dialogue_id
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["dialogue_id"]].append(r)
    for did in grouped:
        grouped[did] = sorted(grouped[did], key=lambda x: x["turn_id"])

    total_turns = 0
    intent_correct = 0
    joint_goal_correct = 0
    complete_correct = 0
    complete_total = 0

    tp = fp = fn = 0
    latencies_ms: List[float] = []

    for dialogue_id, turns in grouped.items():
        sid = orchestrator.create_session(user_id=dialogue_id)

        for t in turns:
            total_turns += 1
            user_text = t["user_text"]
            gold = t["gold"]

            t0 = time.perf_counter()
            pred = orchestrator.process_user_message(sid, user_text)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            pred_intent = norm_text(pred["intent"]["resolved"])
            gold_intent = norm_text(gold.get("intent"))
            if pred_intent == gold_intent:
                intent_correct += 1

            pred_slots = pred.get("state", {}).get("filled_slots", {})
            gold_slots = gold.get("filled_slots", {})

            pset = slots_to_set(pred_slots)
            gset = slots_to_set(gold_slots)

            tp += len(pset & gset)
            fp += len(pset - gset)
            fn += len(gset - pset)

            if pset == gset:
                joint_goal_correct += 1

            if "is_complete" in gold:
                complete_total += 1
                if bool(pred.get("state", {}).get("is_complete", False)) == bool(gold["is_complete"]):
                    complete_correct += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    intent_acc = intent_correct / total_turns if total_turns else 0.0
    jga = joint_goal_correct / total_turns if total_turns else 0.0
    complete_acc = (complete_correct / complete_total) if complete_total else None

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p95_latency = sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)] if latencies_ms else 0.0

    print("\n=== DST EVAL REPORT ===")
    print(f"Total turns: {total_turns}")
    print(f"Intent Accuracy: {intent_acc:.4f}")
    print(f"Slot Precision: {precision:.4f}")
    print(f"Slot Recall:    {recall:.4f}")
    print(f"Slot F1:        {f1:.4f}")
    print(f"Joint Goal Acc: {jga:.4f}")
    if complete_acc is not None:
        print(f"Complete Acc:   {complete_acc:.4f}")
    print(f"Avg latency ms: {avg_latency:.2f}")
    print(f"P95 latency ms: {p95_latency:.2f}")


if __name__ == "__main__":
    main()