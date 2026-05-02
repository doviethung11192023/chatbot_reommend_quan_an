from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
	SentenceTransformer = None

try:
	import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
	lgb = None

try:
	import joblib
except Exception:  # pragma: no cover - optional dependency
	joblib = None

logger = logging.getLogger(__name__)


TIME_SLOT_MAP: Dict[str, List[int]] = {
	"morning": [1, 0, 0, 0],
	"noon": [0, 1, 0, 0],
	"afternoon": [0, 0, 1, 0],
	"evening": [0, 0, 0, 1],
	"all_day": [1, 1, 1, 1],
}


@dataclass
class DBConfig:
	host: str
	port: int
	database: str
	user: str
	password: str
	sslmode: str = "require"

	@classmethod
	def from_env(cls, env: Optional[Dict[str, str]] = None, prefix: str = "DB_") -> "DBConfig":
		env_map = env or os.environ

		def _get(name: str, default: Optional[str] = None, required: bool = True) -> str:
			key = f"{prefix}{name}"
			value = env_map.get(key, default)
			if required and (value is None or value == ""):
				raise ValueError(f"Missing env var: {key}")
			return str(value) if value is not None else ""

		return cls(
			host=_get("HOST"),
			port=int(_get("PORT", default="5432", required=False) or 5432),
			database=_get("NAME"),
			user=_get("USER"),
			password=_get("PASSWORD"),
			sslmode=_get("SSLMODE", default="require", required=False) or "require",
		)


@dataclass
class UserContext:
	lat: float
	lon: float
	user_id: Optional[str] = None
	budget_level: Optional[int] = None
	query_tags: List[str] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"user_id": self.user_id,
			"lat": self.lat,
			"lon": self.lon,
			"budget_level": self.budget_level,
			"query_tags": list(self.query_tags),
		}


@dataclass
class ParsedQuery:
	intent: str
	query_text: str
	food: Optional[str] = None
	price: Optional[int] = None
	distance_km: Optional[float] = None
	raw_slots: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"intent": self.intent,
			"query_text": self.query_text,
			"slots": {
				"food": self.food,
				"price": self.price,
				"distance_km": self.distance_km,
				**self.raw_slots,
			},
		}


@dataclass
class RetrievalSettings:
	top_n: int = 200
	top_k: int = 5
	default_distance_km: float = 5.0
	embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
	use_tfidf: bool = True
	use_embedding: bool = True
	tfidf_weight: float = 0.4
	embedding_weight: float = 0.6
	food_filter_enabled: bool = True
	allow_food_filter_fallback: bool = True
	rerank_enabled: bool = False
	rerank_model_path: Optional[str] = None
	return_debug: bool = False


def build_sql(parsed: ParsedQuery, user_context: UserContext, settings: RetrievalSettings) -> Tuple[str, List[Any]]:
	distance_km = parsed.distance_km or settings.default_distance_km
	distance_threshold = float(distance_km) / 111.0

	price = parsed.price
	price_condition = "TRUE" if price is None else "ABS(r.price_level - %s) <= 1"

	sql = f"""
	SELECT
		r.id,
		r.name,
		r.price_level,
		r.latitude,
		r.longitude,
		SQRT(POWER(r.latitude - %s, 2) + POWER(r.longitude - %s, 2)) AS distance,
		re.content,
		re.embedding,
		COALESCE(AVG(ur.rating), 0) as rating
	FROM restaurants r
	JOIN restaurant_embeddings re ON r.id = re.restaurant_id
	LEFT JOIN user_ratings ur ON r.id = ur.restaurant_id
	GROUP BY r.id, re.content, re.embedding
	HAVING SQRT(POWER(r.latitude - %s, 2) + POWER(r.longitude - %s, 2)) < %s
	   AND {price_condition}
	LIMIT %s
	"""

	params: List[Any] = [
		user_context.lat,
		user_context.lon,
		user_context.lat,
		user_context.lon,
		distance_threshold,
	]
	if price is not None:
		params.append(price)
	params.append(settings.top_n)
	return sql, params


def _parse_embedding(value: Any) -> np.ndarray:
	if value is None:
		return np.array([])
	if isinstance(value, np.ndarray):
		return value
	if isinstance(value, (list, tuple)):
		return np.array(value, dtype=float)
	if isinstance(value, str):
		try:
			return np.array(json.loads(value), dtype=float)
		except Exception:
			return np.array([])
	return np.array([])


def _safe_int(value: Any) -> Optional[int]:
	if value is None:
		return None
	if isinstance(value, (int, np.integer)):
		return int(value)
	try:
		return int(str(value).strip())
	except Exception:
		return None


def _safe_float(value: Any) -> Optional[float]:
	if value is None:
		return None
	if isinstance(value, (float, np.floating)):
		return float(value)
	try:
		return float(str(value).strip())
	except Exception:
		return None


def fetch_candidates(
	conn: psycopg2.extensions.connection,
	parsed: ParsedQuery,
	user_context: UserContext,
	settings: RetrievalSettings,
) -> pd.DataFrame:
	sql, params = build_sql(parsed, user_context, settings)
	df = pd.read_sql(sql, conn, params=params)
	if "embedding" in df.columns:
		df["embedding"] = df["embedding"].apply(_parse_embedding)
	return df


def apply_food_filter(
	df: pd.DataFrame,
	food: Optional[str],
	allow_fallback: bool = True,
) -> pd.DataFrame:
	if not food or "content" not in df.columns or df.empty:
		return df
	original = df
	filtered = df[df["content"].str.contains(food, case=False, na=False)]
	if not filtered.empty:
		return filtered
	return original if allow_fallback else filtered


def compute_tfidf_scores(df: pd.DataFrame, query_text: str) -> np.ndarray:
	if df.empty or "content" not in df.columns:
		return np.array([])
	tfidf = TfidfVectorizer()
	tfidf_matrix = tfidf.fit_transform(df["content"].fillna(""))
	query_vec = tfidf.transform([query_text])
	return cosine_similarity(query_vec, tfidf_matrix).flatten()


def compute_embedding_scores(
	df: pd.DataFrame,
	query_text: str,
	model: Any,
) -> np.ndarray:
	if df.empty or "embedding" not in df.columns:
		return np.array([])
	query_emb = model.encode(query_text)
	scores = [
		cosine_similarity([query_emb], [doc_emb])[0][0]
		if doc_emb is not None and len(doc_emb) > 0
		else 0.0
		for doc_emb in df["embedding"]
	]
	return np.asarray(scores, dtype=float)


def combine_scores(
	df: pd.DataFrame,
	tfidf_scores: Optional[np.ndarray],
	embedding_scores: Optional[np.ndarray],
	settings: RetrievalSettings,
) -> pd.DataFrame:
	output = df.copy()

	if tfidf_scores is not None and len(tfidf_scores) == len(output):
		output["tfidf_score"] = tfidf_scores
	else:
		output["tfidf_score"] = 0.0

	if embedding_scores is not None and len(embedding_scores) == len(output):
		output["embedding_score"] = embedding_scores
	else:
		output["embedding_score"] = 0.0

	output["score"] = (
		settings.tfidf_weight * output["tfidf_score"]
		+ settings.embedding_weight * output["embedding_score"]
	)
	return output


def _listify_tags(tags: Any) -> List[str]:
	if tags is None:
		return []
	if isinstance(tags, list):
		items = tags
	elif isinstance(tags, str):
		try:
			items = json.loads(tags)
			if not isinstance(items, list):
				items = [tags]
		except Exception:
			items = [tags]
	else:
		items = [str(tags)]
	cleaned = [str(t).strip().lower() for t in items if str(t).strip()]
	return sorted(list(set(cleaned)))


def _get_review_counts_batch(
	conn: psycopg2.extensions.connection,
	restaurant_ids: Sequence[int],
) -> Dict[int, int]:
	if not restaurant_ids:
		return {}
	query = """
	SELECT restaurant_id, COUNT(*) as review_count
	FROM user_ratings
	WHERE restaurant_id IN %s
	GROUP BY restaurant_id
	"""
	cur = conn.cursor()
	cur.execute(query, (tuple(restaurant_ids),))
	rows = cur.fetchall()
	cur.close()
	return {int(rid): int(cnt) for rid, cnt in rows}


def _get_tags_batch(
	conn: psycopg2.extensions.connection,
	restaurant_ids: Sequence[int],
) -> Dict[int, List[str]]:
	if not restaurant_ids:
		return {}
	query = """
	SELECT restaurant_id, tags
	FROM menus
	WHERE restaurant_id = ANY(%s)
	"""
	cur = conn.cursor()
	cur.execute(query, (list(restaurant_ids),))
	rows = cur.fetchall()
	cur.close()

	tag_map: Dict[int, List[str]] = {}
	for rid, tags in rows:
		rid_int = int(rid)
		tag_map.setdefault(rid_int, []).extend(_listify_tags(tags))
	tag_map = {rid: sorted(list(set(tags))) for rid, tags in tag_map.items()}
	return tag_map


def _get_open_time_slot_batch(
	conn: psycopg2.extensions.connection,
	restaurant_ids: Sequence[int],
) -> Dict[int, str]:
	if not restaurant_ids:
		return {}
	query = """
	SELECT id, open_time_slot
	FROM restaurants
	WHERE id = ANY(%s)
	"""
	cur = conn.cursor()
	cur.execute(query, (list(restaurant_ids),))
	rows = cur.fetchall()
	cur.close()
	return {int(rid): str(slot) if slot is not None else "" for rid, slot in rows}


def _get_menus_batch(
	conn: psycopg2.extensions.connection,
	restaurant_ids: Sequence[int],
) -> Dict[int, List[str]]:
	if not restaurant_ids:
		return {}
	query = """
	SELECT restaurant_id, dish_name
	FROM menus
	WHERE restaurant_id = ANY(%s)
	"""
	menu_df = pd.read_sql(query, conn, params=(list(restaurant_ids),))
	if menu_df.empty:
		return {}
	grouped = menu_df.groupby("restaurant_id")["dish_name"].apply(list).to_dict()
	return {int(rid): [str(x) for x in names] for rid, names in grouped.items()}


def _get_best_dish_semantic(
	conn: psycopg2.extensions.connection,
	restaurant_ids: Sequence[int],
	food: Optional[str],
	model: Any,
) -> Dict[int, Dict[str, Any]]:
	if not restaurant_ids or not food or model is None:
		return {}
	query = """
	SELECT restaurant_id, dish_name, embedding
	FROM menus
	WHERE restaurant_id = ANY(%s)
	"""
	try:
		menu_df = pd.read_sql(query, conn, params=(list(restaurant_ids),))
	except Exception:
		return {}
	if menu_df.empty or "embedding" not in menu_df.columns:
		return {}
	menu_df["embedding"] = menu_df["embedding"].apply(_parse_embedding)
	query_emb = model.encode(food)

	def _score(emb: np.ndarray) -> float:
		if emb is None or len(emb) == 0:
			return 0.0
		return float(cosine_similarity([query_emb], [emb])[0][0])

	menu_df["score"] = menu_df["embedding"].apply(_score)
	best = menu_df.sort_values("score", ascending=False).groupby("restaurant_id").first()
	out = {}
	for rid, row in best.iterrows():
		out[int(rid)] = {"dish_name": row["dish_name"], "score": float(row["score"])}
	return out


def get_time_context() -> int:
	from datetime import datetime

	hour = datetime.now().hour
	if 6 <= hour < 11:
		return 0
	if 11 <= hour < 14:
		return 1
	if 14 <= hour < 18:
		return 2
	return 3


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	r = 6371.0
	phi1, phi2 = np.radians(lat1), np.radians(lat2)
	dphi = np.radians(lat2 - lat1)
	dlambda = np.radians(lon2 - lon1)
	a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
	return float(2 * r * np.arcsin(np.sqrt(a)))


def time_match_feature(time_context: int, open_time_slot: str) -> int:
	slot_vector = TIME_SLOT_MAP.get(open_time_slot, [0, 0, 0, 0])
	return int(slot_vector[time_context])


def time_soft_match(time_context: int, open_time_slot: str) -> float:
	slot_vector = TIME_SLOT_MAP.get(open_time_slot, [0, 0, 0, 0])
	return 1.0 if slot_vector[time_context] == 1 else 0.2


def safe_log(value: Any) -> float:
	try:
		return float(np.log1p(max(float(value), 0.0)))
	except Exception:
		return 0.0


FEATURE_NAMES: List[str] = [
	"semantic_score",
	"distance_km",
	"distance_score",
	"nearby_bucket",
	"price_level",
	"price_match",
	"price_distance",
	"rating_score",
	"review_score",
	"weighted_rating",
	"popularity_score",
	"time_match",
	"time_soft",
	"open_morning",
	"open_noon",
	"open_afternoon",
	"open_evening",
	"time_context",
	"tag_richness_score",
]


def build_features(user: UserContext, restaurant: Dict[str, Any], global_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
	semantic_score = float(restaurant.get("semantic_score", 0.0))

	distance_km = haversine(
		float(user.lat),
		float(user.lon),
		float(restaurant.get("lat", 0.0)),
		float(restaurant.get("lng", 0.0)),
	)
	distance_score = 1.0 / (1.0 + distance_km)
	if distance_km <= 1:
		nearby_bucket = 3
	elif distance_km <= 3:
		nearby_bucket = 2
	elif distance_km <= 5:
		nearby_bucket = 1
	else:
		nearby_bucket = 0

	price_level = _safe_int(restaurant.get("price_level")) or 0
	user_budget = _safe_int(user.budget_level) or 0
	price_match = 1 if price_level == user_budget else 0
	price_distance = abs(price_level - user_budget)

	rating = float(restaurant.get("rating", 0.0))
	review_count = _safe_int(restaurant.get("review_count")) or 0
	rating_score = rating / 5.0
	review_score = safe_log(review_count)

	avg_rating = float(global_stats.get("avg_rating", 4.0) if global_stats else 4.0)
	m = 50
	weighted_rating = (
		(review_count / (review_count + m)) * rating + (m / (review_count + m)) * avg_rating
	)

	popularity_score = safe_log(review_count)

	open_time_slot = str(restaurant.get("open_time_slot", ""))
	time_context = get_time_context()
	time_match = time_match_feature(time_context, open_time_slot)
	time_soft = time_soft_match(time_context, open_time_slot)
	time_slot_vector = TIME_SLOT_MAP.get(open_time_slot, [0, 0, 0, 0])

	restaurant_tags = restaurant.get("categories", []) or []
	tag_richness_score = safe_log(len(restaurant_tags))

	features = [
		semantic_score,
		distance_km,
		distance_score,
		nearby_bucket,
		price_level,
		price_match,
		price_distance,
		rating_score,
		review_score,
		weighted_rating,
		popularity_score,
		time_match,
		time_soft,
		*time_slot_vector,
		time_context,
		tag_richness_score,
	]
	return np.array(features, dtype=float)


class HybridRetriever:
	def __init__(self, db_config: DBConfig, settings: Optional[RetrievalSettings] = None) -> None:
		self.db_config = db_config
		self.settings = settings or RetrievalSettings()
		self._embedding_model: Any = None
		self._rerank_model: Any = None

	def _connect(self) -> psycopg2.extensions.connection:
		return psycopg2.connect(
			host=self.db_config.host,
			port=self.db_config.port,
			database=self.db_config.database,
			user=self.db_config.user,
			password=self.db_config.password,
			sslmode=self.db_config.sslmode,
		)

	def _get_embedding_model(self) -> Any:
		if not self.settings.use_embedding:
			return None
		if SentenceTransformer is None:
			raise ImportError("sentence_transformers is required for embedding search")
		if self._embedding_model is None:
			self._embedding_model = SentenceTransformer(self.settings.embedding_model_name)
		return self._embedding_model

	def _load_rerank_model(self) -> Any:
		if not self.settings.rerank_enabled or not self.settings.rerank_model_path:
			return None
		if self._rerank_model is not None:
			return self._rerank_model

		path = self.settings.rerank_model_path
		model = None
		if lgb is not None:
			try:
				model = lgb.Booster(model_file=path)
			except Exception:
				model = None
		if model is None and joblib is not None:
			try:
				model = joblib.load(path)
			except Exception:
				model = None
		if model is None:
			logger.warning("Could not load rerank model from %s", path)
		self._rerank_model = model
		return model

	def retrieve(self, parsed: ParsedQuery, user_context: UserContext) -> List[Dict[str, Any]]:
		with self._connect() as conn:
			df = fetch_candidates(conn, parsed, user_context, self.settings)
			if self.settings.food_filter_enabled:
				df = apply_food_filter(df, parsed.food, self.settings.allow_food_filter_fallback)

			if df.empty:
				return []

			tfidf_scores = None
			embedding_scores = None
			if self.settings.use_tfidf:
				tfidf_scores = compute_tfidf_scores(df, parsed.query_text)
			if self.settings.use_embedding:
				embedding_scores = compute_embedding_scores(df, parsed.query_text, self._get_embedding_model())

			scored = combine_scores(df, tfidf_scores, embedding_scores, self.settings)
			scored = scored.sort_values("score", ascending=False).head(self.settings.top_k)

			results = self._to_results(scored)
			self._enrich_results(conn, results, parsed, user_context)
			if self.settings.rerank_enabled:
				results = self._rerank_results(results, user_context)
			return results

	def _to_results(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
		results: List[Dict[str, Any]] = []
		for _, row in df.iterrows():
			result = {
				"id": int(row.get("id")),
				"name": row.get("name"),
				"price_level": _safe_int(row.get("price_level")),
				"lat": _safe_float(row.get("latitude")),
				"lng": _safe_float(row.get("longitude")),
				"distance": _safe_float(row.get("distance")),
				"rating": _safe_float(row.get("rating")),
				"content": row.get("content"),
				"tfidf_score": float(row.get("tfidf_score", 0.0)),
				"embedding_score": float(row.get("embedding_score", 0.0)),
				"score": float(row.get("score", 0.0)),
			}
			result["semantic_score"] = float(result.get("embedding_score") or result.get("score") or 0.0)
			results.append(result)
		return results

	def _enrich_results(
		self,
		conn: psycopg2.extensions.connection,
		results: List[Dict[str, Any]],
		parsed: ParsedQuery,
		user_context: UserContext,
	) -> None:
		if not results:
			return
		restaurant_ids = [r["id"] for r in results]

		review_map = _get_review_counts_batch(conn, restaurant_ids)
		tag_map = _get_tags_batch(conn, restaurant_ids)
		time_map = _get_open_time_slot_batch(conn, restaurant_ids)
		menu_map = _get_menus_batch(conn, restaurant_ids)
		best_dish_map = _get_best_dish_semantic(
			conn,
			restaurant_ids,
			parsed.food,
			self._get_embedding_model() if self.settings.use_embedding else None,
		)

		for r in results:
			rid = r["id"]
			r["review_count"] = int(review_map.get(rid, 0))
			r["categories"] = tag_map.get(rid, [])
			r["open_time_slot"] = time_map.get(rid, "")
			r["menu"] = menu_map.get(rid, [])
			best_dish = best_dish_map.get(rid)
			if best_dish:
				r["best_dish"] = best_dish.get("dish_name")
				r["best_dish_score"] = float(best_dish.get("score", 0.0))

	def _rerank_results(self, results: List[Dict[str, Any]], user_context: UserContext) -> List[Dict[str, Any]]:
		model = self._load_rerank_model()
		if model is None or not results:
			return results

		avg_rating = float(np.mean([r.get("rating", 0.0) or 0.0 for r in results])) if results else 4.0
		features = np.vstack([build_features(user_context, r, {"avg_rating": avg_rating}) for r in results])

		try:
			scores = model.predict(features)
		except Exception:
			return results

		scores = np.asarray(scores, dtype=float)
		ranked_idx = np.argsort(scores)[::-1]
		reranked: List[Dict[str, Any]] = []
		for idx in ranked_idx:
			item = dict(results[int(idx)])
			item["rerank_score"] = float(scores[int(idx)])
			reranked.append(item)
		return reranked


def build_user_context_from_state(
	state: Any,
	default_lat: Optional[float] = None,
	default_lon: Optional[float] = None,
	default_budget: Optional[int] = None,
) -> UserContext:
	context = getattr(state, "context", {}) or {}
	lat = context.get("lat", default_lat)
	lon = context.get("lon", default_lon)
	budget = context.get("budget_level", default_budget)
	if lat is None or lon is None:
		raise ValueError("UserContext requires lat and lon")
	return UserContext(
		user_id=getattr(state, "user_id", None),
		lat=float(lat),
		lon=float(lon),
		budget_level=_safe_int(budget),
	)


def build_parsed_query_from_state(
	state: Any,
	user_text: str,
	default_distance_km: Optional[float] = None,
) -> ParsedQuery:
	slots = getattr(state, "filled_slots", {}) or {}
	slot_map: Dict[str, Any] = {}
	for key, slot in slots.items():
		slot_key = str(key).upper()
		value = getattr(slot, "value", slot)
		slot_map[slot_key] = value

	food = slot_map.get("DISH") or slot_map.get("FOOD")
	price = _safe_int(slot_map.get("PRICE"))
	distance = _safe_float(slot_map.get("DISTANCE_KM"))
	if distance is None:
		distance = _safe_float(slot_map.get("DISTANCE"))
	if distance is None:
		distance = default_distance_km

	intent = getattr(getattr(state, "current_intent", None), "value", None) or ""
	return ParsedQuery(
		intent=intent,
		query_text=user_text,
		food=str(food) if food is not None else None,
		price=price,
		distance_km=distance,
		raw_slots={"slots": slot_map},
	)


def create_default_retriever(settings: Optional[RetrievalSettings] = None) -> HybridRetriever:
	db_config = DBConfig.from_env()
	return HybridRetriever(db_config=db_config, settings=settings)
