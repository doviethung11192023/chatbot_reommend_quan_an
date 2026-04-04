# from __future__ import annotations

# import argparse
# import statistics
# import time
# from typing import Any, Dict, List, Tuple

# import gradio as gr

# from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
# from dialogue_policy.hybrid_policy import HybridPolicy
# from dialogue_policy.ml_policy import SklearnMLPolicy
# from dialogue_policy.rule_based_policy import RuleBasedPolicy
# from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF
# from dialogue_state_tracking.dst import DialogueStateTracker
# import logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# def build_orchestrator(args) -> DialogueOrchestrator:
#     intent_model = IntentClassifierHF(args.intent_model_path)
#     slot_model = SlotExtractorHF(args.slot_model_path)
#     rule_policy = RuleBasedPolicy(rules_path=args.rules_path, debug=True) if args.rules_path else RuleBasedPolicy(debug=True)
#     ml_policy = SklearnMLPolicy(args.ml_policy_path) if args.ml_policy_path else None

#     llm_policy = None
#     if args.use_llm:
#         llm_policy = HuggingFaceLLMPolicy(
#             model_name=args.llm_model,
#             device=args.llm_device,
#             torch_dtype=args.llm_dtype,
#         )

#     policy = HybridPolicy(
#         rule_policy=rule_policy,
#         ml_policy=ml_policy,
#         llm_policy=llm_policy,
#         ml_conf_threshold=args.ml_threshold,
#         debug=True,
#         repeat_window=3,
#         allow_ml_llm_escape=True,
#     )

#     return DialogueOrchestrator(
#         intent_model=intent_model,
#         slot_model=slot_model,
#         dst=DialogueStateTracker(debug=True),
#         policy=policy,
#         debug = True,
#     )


# def format_kpis(stats: Dict[str, Any]) -> str:
#     turns = stats.get("turns", 0)
#     lat = stats.get("latencies_ms", [])
#     complete_hits = stats.get("complete_hits", 0)

#     avg_ms = statistics.mean(lat) if lat else 0.0
#     p95_ms = sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else 0.0
#     complete_rate = (complete_hits / turns * 100.0) if turns > 0 else 0.0

#     return (
#         f"**Turns:** {turns}  \n"
#         f"**Avg latency:** {avg_ms:.1f} ms  \n"
#         f"**P95 latency:** {p95_ms:.1f} ms  \n"
#         f"**State complete rate:** {complete_rate:.1f}%"
#     )


# def main():
#     parser = argparse.ArgumentParser(description="Gradio UI for Hybrid chatbot + DST runtime KPIs")
#     parser.add_argument("--intent_model_path", required=True)
#     parser.add_argument("--slot_model_path", required=True)
#     parser.add_argument("--rules_path", default=None)
#     parser.add_argument("--ml_policy_path", default=None)
#     parser.add_argument("--ml_threshold", type=float, default=0.7)

#     parser.add_argument("--use_llm", action="store_true")
#     parser.add_argument("--llm_model", default="Qwen/Qwen2.5-3B-Instruct")
#     parser.add_argument("--llm_device", default="auto", choices=["auto", "cuda", "cpu"])
#     parser.add_argument("--llm_dtype", default="auto", choices=["auto", "float16", "float32"])

#     parser.add_argument("--server_name", default="0.0.0.0")
#     parser.add_argument("--server_port", type=int, default=7860)
#     parser.add_argument("--share", action="store_true")
#     args = parser.parse_args()

#     orchestrator = build_orchestrator(args)
#     session_id = orchestrator.create_session(user_id="gradio_user")

#     with gr.Blocks(title="Hybrid DST Chatbot") as demo:
#         gr.Markdown("## Hybrid Chatbot (Rule → ML → LLM) + DST Runtime KPIs")

#         chatbot = gr.Chatbot(height=420, type="messages")
#         msg = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: Tìm quán bún bò ở quận 1", lines=2)
#         send = gr.Button("Gửi", variant="primary")
#         clear = gr.Button("Clear")

#         kpis = gr.Markdown("**Turns:** 0")
#         debug = gr.JSON(label="Debug (Intent/Slots/State/Policy)")

#         app_state = gr.State({
#             "session_id": session_id,
#             "turns": 0,
#             "latencies_ms": [],
#             "complete_hits": 0,
#         })

#         def on_send(user_text: str, history: List[Dict[str, str]], st: Dict[str, Any]):
#             user_text = (user_text or "").strip()
#             if not user_text:
#                 return history, st, format_kpis(st), {}

#             t0 = time.perf_counter()
#             result = orchestrator.process_user_message(st["session_id"], user_text)
#             latency_ms = (time.perf_counter() - t0) * 1000.0

#             bot_text = result["action"]["template"]
#             intent = result["intent"]["resolved"]
#             action = result["action"]["type"]
#             slots = result["slots"]

#             answer = f"{bot_text}\n\n(intent={intent}, action={action}, slots={slots})"

#             history = history + [
#                 {"role": "user", "content": user_text},
#                 {"role": "assistant", "content": answer},
#             ]

#             st["turns"] += 1
#             st["latencies_ms"].append(latency_ms)
#             if result.get("state", {}).get("is_complete", False):
#                 st["complete_hits"] += 1

#             return history, st, format_kpis(st), result

#         def on_clear():
#             new_session = orchestrator.create_session(user_id="gradio_user")
#             st = {"session_id": new_session, "turns": 0, "latencies_ms": [], "complete_hits": 0}
#             return [], st, format_kpis(st), {}

#         send.click(on_send, [msg, chatbot, app_state], [chatbot, app_state, kpis, debug])
#         msg.submit(on_send, [msg, chatbot, app_state], [chatbot, app_state, kpis, debug])
#         clear.click(on_clear, outputs=[chatbot, app_state, kpis, debug])

#     demo.queue().launch(
#         server_name=args.server_name,
#         server_port=args.server_port,
#         share=args.share,
#     )


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Tuple

import gradio as gr

from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.ml_policy import SklearnMLPolicy
from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF
from dialogue_state_tracking.dst import DialogueStateTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_orchestrator(args) -> DialogueOrchestrator:
    intent_classifier = IntentClassifierHF(
        # model_name=args.intent_model,
        # debug=args.debug,
        args.intent_model
        ,device="cuda"
    )
    slot_extractor = SlotExtractorHF(
        # model_name=args.slot_model,
        # debug=args.debug,
        args.slot_model,
        device="cuda"
    )

    rule_policy = RuleBasedPolicy()
    # ml_policy = SklearnMLPolicy(model_path=args.ml_model, debug=args.debug) if args.ml_model else None
    # llm_policy = HuggingFaceLLMPolicy(model_name=args.llm_model, debug=args.debug) if args.llm_model else None
    ml_policy = SklearnMLPolicy(args.ml_model) if args.ml_model else None

    llm_policy = None
   
    llm_policy = HuggingFaceLLMPolicy(
            model_name=args.llm_model,
            device=args.llm_device,
            torch_dtype=args.llm_dtype,
        )
    dst = DialogueStateTracker(debug=args.debug)

    policy = HybridPolicy(
        rule_policy=rule_policy,
        ml_policy=ml_policy,
        llm_policy=llm_policy,
        ml_conf_threshold=args.ml_threshold,
        debug=args.debug,
        repeat_window=args.repeat_window,
        allow_ml_llm_escape=True,
        state_quality_threshold=args.state_quality_threshold,
    )

    orchestrator = DialogueOrchestrator(
        intent_model=intent_classifier,
        slot_model=slot_extractor,
        dst=dst,
        policy=policy,
        debug=True,
    )

    return orchestrator


def format_kpis(stats: Dict[str, Any]) -> str:
    state_quality = stats.get("state_quality", None)
    policy_source = stats.get("policy_source", "N/A")
    intent = stats.get("resolved_intent", "N/A")
    missing_slots = stats.get("missing_slots", [])
    slot_conflicts = stats.get("slot_conflicts", 0)

    return (
        f"### Debug Summary\n"
        f"- **Intent**: `{intent}`\n"
        f"- **Policy source**: `{policy_source}`\n"
        f"- **State quality**: `{state_quality:.3f}`\n" if isinstance(state_quality, (float, int)) else f"- **State quality**: `N/A`\n"
    ) + (
        f"- **Missing slots**: `{missing_slots}`\n"
        f"- **Slot conflicts**: `{slot_conflicts}`\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-model", type=str, default="models/intent_classifier")
    parser.add_argument("--slot-model", type=str, default="models/slot_extractor")
    parser.add_argument("--ml-model", type=str, default="")
    parser.add_argument("--llm-model", type=str, default="")
    parser.add_argument("--ml-threshold", type=float, default=0.7)
    parser.add_argument("--repeat-window", type=int, default=3)
    parser.add_argument("--state-quality-threshold", type=float, default=0.65)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--llm_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--llm_dtype", default="auto", choices=["auto", "float16", "float32"])
    args = parser.parse_args()

    orchestrator = build_orchestrator(args)
    logger.info(
        "[App] started debug=%s state_quality_threshold=%.2f ml_threshold=%.2f repeat_window=%d",
        args.debug,
        args.state_quality_threshold,
        args.ml_threshold,
        args.repeat_window,
    )

    session_id = None

    def respond(user_text: str, history: List[Tuple[str, str]]):
        nonlocal session_id
        if session_id is None:
            session_id = orchestrator.dst.create_session()

        result = orchestrator.process_user_message(session_id=session_id, user_text=user_text)
        state = result.get("state", {})
        action = result.get("action", {})
        policy = result.get("policy", {})

        state_quality = state.get("state_quality")
        if state_quality is None:
            state_obj = orchestrator.dst.get_state(session_id)
            if state_obj and hasattr(state_obj, "get_state_quality"):
                state_quality = state_obj.get_state_quality()

        logger.info(
            "[App] session_id=%s intent=%s action=%s policy=%s state_quality=%s state=%s",
            session_id,
            result.get("intent", {}).get("resolved"),
            action.get("type"),
            policy.get("source"),
            state_quality,
            state,
        )

        bot_text = action.get("template", "")
        debug_panel = format_kpis(
            {
                "resolved_intent": result.get("intent", {}).get("resolved"),
                "policy_source": policy.get("source"),
                "state_quality": state_quality,
                "missing_slots": state.get("missing_slots", []),
                "slot_conflicts": state.get("slot_conflicts", 0),
            }
        )

        history = history + [(user_text, bot_text)]
        return history, debug_panel

    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot Recommend - Hybrid Debug")

        chatbot = gr.Chatbot(label="Chat")
        debug_box = gr.Markdown(label="Debug")

        msg = gr.Textbox(label="Nhập tin nhắn", placeholder="Nhập câu hỏi...")
        send = gr.Button("Gửi")

        def on_send(user_text, history):
            return respond(user_text, history)

        send.click(on_send, inputs=[msg, chatbot], outputs=[chatbot, debug_box])
        msg.submit(on_send, inputs=[msg, chatbot], outputs=[chatbot, debug_box])

    demo.launch()


if __name__ == "__main__":
    main()