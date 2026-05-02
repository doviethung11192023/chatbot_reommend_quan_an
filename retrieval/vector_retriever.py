from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from retrieval.hybrid_retriever import compute_embedding_scores

try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
	SentenceTransformer = None


class VectorRetriever:
	def __init__(self, model_name: str) -> None:
		self.model_name = model_name
		self._model: Any = None

	def _get_model(self) -> Any:
		if SentenceTransformer is None:
			raise ImportError("sentence_transformers is required for embedding search")
		if self._model is None:
			self._model = SentenceTransformer(self.model_name)
		return self._model

	def score(self, df: pd.DataFrame, query_text: str):
		model = self._get_model()
		return compute_embedding_scores(df, query_text, model)
