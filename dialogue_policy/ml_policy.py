from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from dialogue_state_tracking.state_schema import DialogueState


@dataclass
class MLPrediction:
    action: str
    confidence: float


class SklearnMLPolicy:
    """
    Bundle format (.joblib):
      {
        "vectorizer": CountVectorizer/TfidfVectorizer,
        "model": sklearn classifier
      }
    """

    def __init__(self, model_bundle_path: str):
        import joblib

        bundle = joblib.load(model_bundle_path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]

    @staticmethod
    def state_to_text(state: DialogueState) -> str:
        intent = state.current_intent.name if state.current_intent else "NONE"
        missing = ",".join(sorted(state.get_missing_slots()))
        filled = ",".join(
            sorted([f"{k}:{v.value}" for k, v in state.filled_slots.items()])
        )
        turn_count = len(state.turns)
        last_utt = state.turns[-1].user_utterance if state.turns else ""
        return (
            f"intent={intent} | missing={missing} | filled={filled} "
            f"| turns={turn_count} | last={last_utt}"
        )

    def predict_action(self, state: DialogueState) -> Dict[str, Any]:
        text = self.state_to_text(state)
        x = self.vectorizer.transform([text])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            idx = int(probs.argmax())
            return {
                "action": str(self.model.classes_[idx]).upper(),
                "confidence": float(probs[idx]),
            }

        pred = str(self.model.predict(x)[0]).upper()
        return {"action": pred, "confidence": 1.0}