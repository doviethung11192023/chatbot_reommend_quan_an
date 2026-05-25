from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import psycopg2

from retrieval.hybrid_retriever import (
    DBConfig,
    ParsedQuery,
    RetrievalSettings,
    UserContext,
    build_sql,
    fetch_candidates,
)


class SQLRetriever:
    def __init__(self, db_config: DBConfig, settings: Optional[RetrievalSettings] = None) -> None:
        self.db_config = db_config
        self.settings = settings or RetrievalSettings()

    def _connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.user,
            password=self.db_config.password,
            sslmode=self.db_config.sslmode,
        )

    def build_query(self, parsed: ParsedQuery, user_context: UserContext) -> tuple[str, List[Any]]:
        return build_sql(parsed, user_context, self.settings)

    def fetch(self, parsed: ParsedQuery, user_context: UserContext):
        with self._connect() as conn:
            return fetch_candidates(conn, parsed, user_context, self.settings)

    def filter_by_candidates(
        self,
        candidate_ids: Sequence[int],
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        price: Optional[int] = None,
        time_slot: Optional[str] = None,
        distance_km: float = 5.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Filter candidate restaurants by spatial distance, price, and time slot.

        Uses haversine formula for accurate distance calculation on PostgreSQL.

        Args:
            candidate_ids: list of restaurant IDs to filter.
            lat: user latitude.
            lng: user longitude.
            price: max price_level to accept.
            time_slot: required open_time_slot value.
            distance_km: max distance in km.
            limit: max results.

        Returns:
            List of dicts with id, name, price_level, latitude, longitude, distance.
        """
        if not candidate_ids:
            return []

        ids_list = [int(x) for x in candidate_ids]

        query = """
            SELECT r.id, r.name, r.price_level,
                   r.latitude, r.longitude,
                   (6371 * acos(
                       cos(radians(%s)) * cos(radians(latitude)) *
                       cos(radians(longitude) - radians(%s)) +
                       sin(radians(%s)) * sin(radians(latitude))
                   )) AS distance
            FROM restaurants r
            WHERE r.id = ANY(%s)
        """

        params: List[Any] = [lat, lng, lat, ids_list]

        if price is not None:
            query += " AND r.price_level <= %s"
            params.append(price)

        if time_slot is not None:
            query += " AND r.open_time_slot = %s"
            params.append(time_slot)

        if lat is not None and lng is not None:
            query += " HAVING (6371 * acos(" \
                     "cos(radians(%s)) * cos(radians(latitude)) * " \
                     "cos(radians(longitude) - radians(%s)) + " \
                     "sin(radians(%s)) * sin(radians(latitude)))) <= %s"
            params.extend([float(lat), float(lng), float(lat), float(distance_km)])

        query += " ORDER BY distance LIMIT %s"
        params.append(limit)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            {
                "id": int(row[0]),
                "name": row[1],
                "price_level": int(row[2]) if row[2] is not None else None,
                "lat": float(row[3]) if row[3] is not None else None,
                "lng": float(row[4]) if row[4] is not None else None,
                "distance": float(row[5]) if row[5] is not None else None,
            }
            for row in rows
        ]
