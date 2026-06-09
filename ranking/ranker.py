from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from retrieval.hybrid_retriever import UserContext, build_features

try:
	import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
	lgb = None

try:
	import joblib
except Exception:  # pragma: no cover - optional dependency
	joblib = None

logger = logging.getLogger(__name__)


class LightGBMReranker:
	def __init__(self, model_path: str) -> None:
		self.model_path = model_path
		self._model: Any = None

	def _load(self) -> Any:
		if self._model is not None:
			return self._model
		model = None
		if lgb is not None:
			try:
				model = lgb.Booster(model_file=self.model_path)
			except Exception:
				model = None
		if model is None and joblib is not None:
			try:
				model = joblib.load(self.model_path)
			except Exception:
				model = None
		if model is None:
			logger.warning("Could not load model from %s", self.model_path)
		self._model = model
		return model

	def rerank(self, results: List[Dict[str, Any]], user_context: UserContext) -> List[Dict[str, Any]]:
		if not results:
			return results
		model = self._load()
		if model is None:
			return results

		avg_rating = float(np.mean([r.get("rating", 0.0) or 0.0 for r in results]))
		features = np.vstack([build_features(user_context, r, {"avg_rating": avg_rating}) for r in results])
		try:
			scores = model.predict(features)
		except Exception:
			return results

		scores = np.asarray(scores, dtype=float)
		ranked_idx = np.argsort(scores)[::-1]
		reranked = []
		for idx in ranked_idx:
			item = dict(results[int(idx)])
			item["rerank_score"] = float(scores[int(idx)])
			reranked.append(item)
		return reranked
