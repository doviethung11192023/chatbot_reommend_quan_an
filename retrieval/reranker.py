from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import psycopg2

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


class CrossEncoderReranker:
    """CrossEncoder-based reranker for restaurant search results."""

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        self._conn = conn
        self.model_name = model_name
        self._model: Any = None

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    def _get_model(self) -> Any:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoder reranker"
            )
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _fetch_menu_text(self, restaurant_ids: Sequence[int]) -> Dict[int, str]:
        """Fetch concatenated dish_name + description per restaurant_id."""
        if not restaurant_ids:
            return {}

        ids_list = [int(rid) for rid in restaurant_ids]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.restaurant_id, m.dish_name, m.description
                FROM menus m
                WHERE m.restaurant_id = ANY(%s)
                """,
                (ids_list,),
            )
            rows = cur.fetchall()

        text_map: Dict[int, str] = {}
        for rid, dish, desc in rows:
            rid_int = int(rid)
            text_map.setdefault(rid_int, "")
            text_map[rid_int] += f"{dish}. {desc or ''}. "
        return text_map

    def rerank(
        self,
        query: str,
        rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using CrossEncoder.

        Args:
            query: user query text.
            rows: list of candidate dicts with at minimum 'id' (restaurant_id).

        Returns:
            Candidates sorted by CrossEncoder score descending, with 'score' updated.
        """
        if not rows:
            return []

        text_map = self._fetch_menu_text([r.get("id") for r in rows if r.get("id") is not None])

        pairs: List[Tuple[str, str]] = []
        filtered: List[Dict[str, Any]] = []
        for r in rows:
            rid = r.get("id")
            if rid is None:
                continue
            text = text_map.get(int(rid), "")
            pairs.append((query, text))
            filtered.append(r)

        if not pairs:
            return list(rows)

        try:
            model = self._get_model()
            scores = model.predict(pairs)
        except Exception:
            logger.warning("CrossEncoder rerank failed, returning original order")
            return list(rows)

        for i, s in enumerate(scores):
            filtered[i]["rerank_score"] = float(s)
            filtered[i]["score"] = float(s)

        filtered.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
        return filtered
