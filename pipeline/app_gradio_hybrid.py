 


from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Tuple

import gradio as gr

from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF
from dialogue_state_tracking.dst import DialogueStateTracker
from retrieval.hybrid_retriever import create_default_retriever, RetrievalSettings

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
    # Option 3: ML policy is intentionally disabled in runtime decision path.
    ml_policy = None

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

    retriever = None
    try:
        settings = RetrievalSettings(
            top_k=args.retrieval_top_k,
            top_n=args.retrieval_top_n,
            rerank_enabled=bool(args.rerank_model_path),
            rerank_model_path=args.rerank_model_path or None,
        )
        retriever = create_default_retriever(settings=settings)
    except Exception as exc:
        logger.warning("Retriever disabled: %s", exc)

    orchestrator = DialogueOrchestrator(
        intent_model=intent_classifier,
        slot_model=slot_extractor,
        dst=dst,
        policy=policy,
        debug=True,
        retriever=retriever,
        retrieval_default_lat=args.default_lat,
        retrieval_default_lon=args.default_lon,
        retrieval_default_budget=args.default_budget,
        retrieval_default_distance_km=args.default_distance_km,
    )

    return orchestrator


def format_kpis(stats: Dict[str, Any]) -> str:
    state_quality = stats.get("state_quality", None)
    policy_source = stats.get("policy_source", "N/A")
    intent = stats.get("resolved_intent", "N/A")
    missing_slots = stats.get("missing_slots", [])
    slot_conflicts = stats.get("slot_conflicts", 0)
    policy_plan = stats.get("policy_plan")

    kpi_text = (
        f"### Debug Summary\n"
        f"- **Intent**: `{intent}`\n"
        f"- **Policy source**: `{policy_source}`\n"
        f"- **State quality**: `{state_quality:.3f}`\n" if isinstance(state_quality, (float, int)) else f"- **State quality**: `N/A`\n"
    ) + (
        f"- **Missing slots**: `{missing_slots}`\n"
        f"- **Slot conflicts**: `{slot_conflicts}`\n"
    )

    if policy_plan:
        kpi_text += f"- **Policy plan**: `{policy_plan}`\n"

    return kpi_text


def format_recommendations(recs: List[Dict[str, Any]]) -> str:
    if not recs:
        return ""
    lines = ["Top recommendations:"]
    for idx, item in enumerate(recs, start=1):
        name = item.get("name") or "(unknown)"
        rating = item.get("rating")
        best_dish = item.get("best_dish")
        line = f"{idx}. {name}"
        if isinstance(rating, (float, int)):
            line += f" | rating={rating:.1f}"
        if best_dish:
            line += f" | dish={best_dish}"
        lines.append(line)
    return "\n".join(lines)


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
    parser.add_argument("--default-lat", type=float, default=None)
    parser.add_argument("--default-lon", type=float, default=None)
    parser.add_argument("--default-budget", type=int, default=None)
    parser.add_argument("--default-distance-km", type=float, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--retrieval-top-n", type=int, default=200)
    parser.add_argument("--rerank-model-path", type=str, default="")
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
        recs = result.get("recommendations") or []

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
        rec_block = format_recommendations(recs)
        if rec_block:
            bot_text = bot_text + "\n\n" + rec_block
        debug_panel = format_kpis(
            {
                "resolved_intent": result.get("intent", {}).get("resolved"),
                "policy_source": policy.get("source"),
                "state_quality": state_quality,
                "missing_slots": state.get("missing_slots", []),
                "slot_conflicts": state.get("slot_conflicts", 0),
                "policy_plan": state.get("policy_plan"),
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