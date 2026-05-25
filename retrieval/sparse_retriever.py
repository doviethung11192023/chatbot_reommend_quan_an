from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

logger = logging.getLogger(__name__)

STOPWORDS = {
    "quán", "gần", "tôi", "ở", "có", "không", "muốn", "ăn", "món",
    "nào", "gì", "cho", "và", "ở đâu", "tim", "tìm", "kiếm",
    "một", "vài", "các", "những", "với", "hay", "nha", "nhe",
}


class SparseRetriever:
    """PostgreSQL Full-Text Search retriever using tsvector on menus table."""

    def __init__(self, conn: psycopg2.extensions.connection):
        self._conn = conn

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    def _clean_query(self, query: str) -> str:
        tokens = query.lower().split()
        tokens = [t for t in tokens if t not in STOPWORDS]
        return " ".join(tokens)

    def search(
        self,
        query: str,
        slots: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search menus via PostgreSQL full-text search.

        Returns list of dicts with keys: menu_id, restaurant_id, dish_name, score.
        """
        slots = slots or {}
        queries: List[str] = []

        dish = slots.get("dish") or slots.get("DISH")
        if dish and str(dish).strip():
            queries.append(str(dish).strip())

        all_results: List[Tuple[int, int, str, float]] = []

        for q in queries:
            q_clean = self._clean_query(q)
            if not q_clean:
                continue
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (restaurant_id)
                           id AS menu_id,
                           restaurant_id,
                           dish_name,
                           ts_rank(tsv, websearch_to_tsquery(%s)) AS score
                    FROM menus
                    WHERE tsv @@ websearch_to_tsquery(%s)
                    ORDER BY restaurant_id, score DESC
                    LIMIT %s
                    """,
                    (q_clean, q_clean, top_k),
                )
                rows = cur.fetchall()
                all_results.extend(rows)

        if not all_results:
            return self._fallback_like_search(query, top_k)

        result_map: Dict[int, Dict[str, Any]] = {}
        for menu_id, rid, dish_name, score in all_results:
            if rid not in result_map or float(score) > float(result_map[rid]["score"]):
                result_map[rid] = {
                    "menu_id": menu_id,
                    "restaurant_id": int(rid),
                    "dish_name": dish_name,
                    "score": float(score),
                }

        ranked = sorted(result_map.values(), key=lambda x: x["score"], reverse=True)
        return ranked

    def _fallback_like_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_clean = self._clean_query(query)
        if not q_clean:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id AS menu_id, restaurant_id, dish_name
                FROM menus
                WHERE dish_name ILIKE %s
                LIMIT %s
                """,
                (f"%{q_clean}%", top_k),
            )
            rows = cur.fetchall()

        return [
            {
                "menu_id": menu_id,
                "restaurant_id": int(rid),
                "dish_name": dish_name,
                "score": 0.1,
            }
            for menu_id, rid, dish_name in rows
        ]
