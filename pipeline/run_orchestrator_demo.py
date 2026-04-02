import argparse

from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import (
    DialogueOrchestrator,
    IntentClassifierHF,
    SlotExtractorHF,
)


def main():
    parser = argparse.ArgumentParser(description="Interactive DST + Policy demo")
    parser.add_argument("--intent_model_path", required=True, help="Path intent model")
    parser.add_argument("--slot_model_path", required=True, help="Path slot model")
    parser.add_argument("--rules_path", default=None, help="Path policy_rules.json (optional)")
    parser.add_argument("--user_id", default=None)
    args = parser.parse_args()

    intent_model = IntentClassifierHF(args.intent_model_path)
    slot_model = SlotExtractorHF(args.slot_model_path)
    policy = RuleBasedPolicy(rules_path=args.rules_path) if args.rules_path else RuleBasedPolicy()

    orchestrator = DialogueOrchestrator(
        intent_model=intent_model,
        slot_model=slot_model,
        policy=policy,
    )
    session_id = orchestrator.create_session(user_id=args.user_id)

    print(f"✅ Session: {session_id}")
    print("Nhập 'quit' để thoát.\n")

    # while True:
    #     text = input("🧑 User: ").strip()
    #     if text.lower() in {"quit", "exit", "q"}:
    #         print("👋 Bye")
    #         break
    #     if not text:
    #         continue

    #     result = orchestrator.process_user_message(session_id, text)

    #     print(f"🔎 Intent raw/resolved: {result['intent']['raw']} -> {result['intent']['resolved']}")
    #     print(f"🧩 Slots: {result['slots']}")
    #     print(f"🤖 Action: {result['action']['type']} ({result['action']['slot']})")
    #     print(f"💬 Bot: {result['action']['template']}\n")
    test_inputs = [
    "Tôi muốn ăn bún bò",
    "Quán nào giá rẻ?",
    "Ở quận 1 có không?"
    ]

    for text in test_inputs:
        print(f"\n🧑 User: {text}")

        result = orchestrator.process_user_message(session_id, text)
       

        print(f"🔎 Intent: {result['intent']['resolved']}")
        print(f"🧩 Slots: {result['slots']}")
        print(f"🤖 Bot: {result['action']['template']}")


if __name__ == "__main__":
    main()