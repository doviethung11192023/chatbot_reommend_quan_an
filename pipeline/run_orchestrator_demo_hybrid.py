import argparse

from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.ml_policy import SklearnMLPolicy
from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF


def main():
    parser = argparse.ArgumentParser(description="Hybrid demo: Rule -> ML -> LLM")
    parser.add_argument("--intent_model_path", required=True)
    parser.add_argument("--slot_model_path", required=True)
    parser.add_argument("--rules_path", default=None)
    parser.add_argument("--ml_policy_path", default=None)
    parser.add_argument("--ml_threshold", type=float, default=0.7)

    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--llm_dtype", default="auto", choices=["auto", "float16", "float32"])

    parser.add_argument("--user_id", default=None)
    args = parser.parse_args()

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
    )

    orchestrator = DialogueOrchestrator(
        intent_model=intent_model,
        slot_model=slot_model,
        policy=policy,
    )

    session_id = orchestrator.create_session(user_id=args.user_id)
    print(f"✅ Session: {session_id}")
    print("Nhập 'quit' để thoát, 'log' để xem policy source.\n")

    while True:
        text = input("🧑 User: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("👋 Bye")
            break
        if not text:
            continue

        result = orchestrator.process_user_message(session_id, text)

        print(f"🔎 Intent: {result['intent']['raw']} -> {result['intent']['resolved']} ({result['intent']['confidence']:.2f})")
        print(f"🧩 Slots: {result['slots']}")
        print(f"🤖 Action: {result['action']['type']} ({result['action']['slot']})")
        print(f"💬 Bot: {result['action']['template']}")

        if "policy" in result:
            print(f"⚙️ Policy: {result['policy']}")
        print()


if __name__ == "__main__":
    main()