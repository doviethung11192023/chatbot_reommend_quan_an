from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, List, Tuple

import gradio as gr

from dialogue_policy.hf_llm_policy import HuggingFaceLLMPolicy
from dialogue_policy.hybrid_policy import HybridPolicy
from dialogue_policy.ml_policy import SklearnMLPolicy
from dialogue_policy.rule_based_policy import RuleBasedPolicy
from pipeline.dialogue_manager import DialogueOrchestrator, IntentClassifierHF, SlotExtractorHF


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
    )

    return DialogueOrchestrator(
        intent_model=intent_model,
        slot_model=slot_model,
        policy=policy,
    )


def format_kpis(stats: Dict[str, Any]) -> str:
    turns = stats.get("turns", 0)
    lat = stats.get("latencies_ms", [])
    complete_hits = stats.get("complete_hits", 0)

    avg_ms = statistics.mean(lat) if lat else 0.0
    p95_ms = sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else 0.0
    complete_rate = (complete_hits / turns * 100.0) if turns > 0 else 0.0

    return (
        f"**Turns:** {turns}  \n"
        f"**Avg latency:** {avg_ms:.1f} ms  \n"
        f"**P95 latency:** {p95_ms:.1f} ms  \n"
        f"**State complete rate:** {complete_rate:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Hybrid chatbot + DST runtime KPIs")
    parser.add_argument("--intent_model_path", required=True)
    parser.add_argument("--slot_model_path", required=True)
    parser.add_argument("--rules_path", default=None)
    parser.add_argument("--ml_policy_path", default=None)
    parser.add_argument("--ml_threshold", type=float, default=0.7)

    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--llm_dtype", default="auto", choices=["auto", "float16", "float32"])

    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    orchestrator = build_orchestrator(args)
    session_id = orchestrator.create_session(user_id="gradio_user")

    with gr.Blocks(title="Hybrid DST Chatbot") as demo:
        gr.Markdown("## Hybrid Chatbot (Rule → ML → LLM) + DST Runtime KPIs")

        chatbot = gr.Chatbot(height=420, type="messages")
        msg = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: Tìm quán bún bò ở quận 1", lines=2)
        send = gr.Button("Gửi", variant="primary")
        clear = gr.Button("Clear")

        kpis = gr.Markdown("**Turns:** 0")
        debug = gr.JSON(label="Debug (Intent/Slots/State/Policy)")

        app_state = gr.State({
            "session_id": session_id,
            "turns": 0,
            "latencies_ms": [],
            "complete_hits": 0,
        })

        def on_send(user_text: str, history: List[Dict[str, str]], st: Dict[str, Any]):
            user_text = (user_text or "").strip()
            if not user_text:
                return history, st, format_kpis(st), {}

            t0 = time.perf_counter()
            result = orchestrator.process_user_message(st["session_id"], user_text)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            bot_text = result["action"]["template"]
            intent = result["intent"]["resolved"]
            action = result["action"]["type"]
            slots = result["slots"]

            answer = f"{bot_text}\n\n(intent={intent}, action={action}, slots={slots})"

            history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]

            st["turns"] += 1
            st["latencies_ms"].append(latency_ms)
            if result.get("state", {}).get("is_complete", False):
                st["complete_hits"] += 1

            return history, st, format_kpis(st), result

        def on_clear():
            new_session = orchestrator.create_session(user_id="gradio_user")
            st = {"session_id": new_session, "turns": 0, "latencies_ms": [], "complete_hits": 0}
            return [], st, format_kpis(st), {}

        send.click(on_send, [msg, chatbot, app_state], [chatbot, app_state, kpis, debug])
        msg.submit(on_send, [msg, chatbot, app_state], [chatbot, app_state, kpis, debug])
        clear.click(on_clear, outputs=[chatbot, app_state, kpis, debug])

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
