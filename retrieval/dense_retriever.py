from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import psycopg2

logger = logging.getLogger(__name__)


class DenseRetriever:
    """pgvector-based dense retrieval using embedding <-> operator on menus table."""

    def __init__(self, conn: psycopg2.extensions.connection):
        self._conn = conn

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    def search(
        self,
        query_embedding: np.ndarray,
        candidate_ids: Optional[List[int]] = None,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Search menus by embedding distance via pgvector.

        Args:
            query_embedding: numpy array of the query embedding.
            candidate_ids: optional list of restaurant_ids to restrict search.
            top_k: number of results to return.

        Returns:
            List of (restaurant_id, distance) sorted by distance ascending.
        """
        query_vec = [float(x) for x in query_embedding]

        with self.conn.cursor() as cur:
            if candidate_ids:
                cur.execute(
                    """
                    SELECT restaurant_id,
                           MIN(embedding <-> %s::vector) AS distance
                    FROM menus
                    WHERE restaurant_id = ANY(%s)
                    GROUP BY restaurant_id
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_vec, candidate_ids, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT restaurant_id,
                           MIN(embedding <-> %s::vector) AS distance
                    FROM menus
                    GROUP BY restaurant_id
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_vec, top_k),
                )

            rows = cur.fetchall()

        return [(int(rid), float(distance)) for rid, distance in rows]
