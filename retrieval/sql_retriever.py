from __future__ import annotations

from typing import Any, List, Optional

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
