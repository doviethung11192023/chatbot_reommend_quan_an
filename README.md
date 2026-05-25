# Chatbot Recommend (Option 3)

Dự án chatbot gợi ý quán ăn với kiến trúc Option 3: **LLM là người ra quyết định, Rule là lớp an toàn, DST là bộ nhớ hội thoại**.

Mục tiêu: Gợi ý quán ăn theo món, khu vực, giá cả, và sở thích người dùng.

---

## Kiến trúc Tổng thể

```
User Input
    │
    ▼
┌─────────────────────────┐
│ 1. Intent Classification │  IntentClassifierHF (HF AutoModelForSequenceClassification, 12 classes)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. Domain Gate           │  DomainGate — lọc cancel/goodbye/small-talk/out-of-scope (regex)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. Slot Extraction       │  SlotExtractorHF (HF AutoModelForTokenClassification, BIO tagging)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. Intent Resolution     │  Context-aware: follow-up merging, low-confidence fallback
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│ 5. DST Update (DialogueStateTracker)             │
│    a. SlotValidator — lọc noise, chuẩn hóa       │
│    b. IntentShiftDetector — cancel/goodbye/change │
│    c. SemanticSlotRanker — merge slot cũ vs mới   │
│    d. State Commit — lưu turn, cập nhật state     │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│ 6. Policy Decision (HybridPolicy)                │
│    a. Dialogue act handling (CANCEL/GOODBYE/...) │
│    b. Safety rule layer (RuleBasedPolicy, 6 rules)│
│    c. LLM decision (HuggingFaceLLMPolicy, Qwen2.5)│
│    d. State quality guards (downgrade/upgrade)    │
│    e. Template selection (chống lặp)              │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│ 7. Retrieval (HybridRetriever)                   │
│    [chỉ chạy khi action=RECOMMEND + state đủ]    │
│    a. Sparse Search (PostgreSQL FTS trên menus)  │
│    b. Dense Search (pgvector ANN trên menus)     │
│    c. RRF Fusion (Reciprocal Rank Fusion)        │
│    d. SQL Filter (spatial/price/time, restaurants)│
│    e. CrossEncoder Rerank (BAAI/bge-reranker-base)│
│    f. Enrich (reviews, tags, menus, best_dish)   │
│    g. LightGBM Rerank (optional, 19 features)    │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│ 8. Bot Response          │  Template selection + recommendation list
└─────────────────────────┘
```

---

## Pipeline Chi tiết

### Bước 1: Intent Classification

- **File**: `pipeline/dialogue_manager.py` → `IntentClassifierHF`
- **Model**: HuggingFace `AutoModelForSequenceClassification`
- **12 intent types**:

| ID | Intent | Mô tả | Ví dụ |
|----|--------|-------|-------|
| 0 | ASK_DIRECTION | Hỏi đường đi | "Chỉ đường đến quán X" |
| 1 | ASK_FOOD_TYPE | Hỏi loại món ăn | "Có món chay không?" |
| 2 | ASK_LOCATION | Hỏi địa điểm | "Quán này ở đâu?" |
| 3 | ASK_OPEN_TIME | Hỏi giờ mở cửa | "Mấy giờ mở cửa?" |
| 4 | ASK_PRICE | Hỏi giá | "Giá bao nhiêu?" |
| 5 | ASK_REVIEW | Hỏi đánh giá | "Quán này ngon không?" |
| 6 | COMPARE_PLACES | So sánh quán | "Quán A với quán B?" |
| 7 | NO_CLEAR_INTENT | Không rõ ý định | "không biết nữa" |
| 8 | OUT_OF_SCOPE | Ngoài phạm vi | "thời tiết hôm nay" |
| 9 | RECOMMEND_FOOD | Gợi ý món ăn | "Món gì ngon?" |
| 10 | RECOMMEND_PLACE_NEARBY | Gợi ý quán gần đây | "Tìm quán phở gần đây" |
| 11 | SMALL_TALK | Chào hỏi | "xin chào", "cảm ơn" |

### Bước 2: Domain Gate

- **File**: `pipeline/domain_gate.py` → `DomainGate`
- Lọc các pattern: CANCEL, GOODBYE, SMALL_TALK, OUT_OF_SCOPE, CHANGE cues
- Có thể suppress slot extraction cho intent ngoài domain

### Bước 3: Slot Extraction

- **File**: `pipeline/dialogue_manager.py` → `SlotExtractorHF`
- **Model**: HuggingFace `AutoModelForTokenClassification` (NER, BIO tagging)
- **Slot types**: DISH (món ăn), LOCATION (địa điểm), PRICE (giá), TASTE (vị), TIME (thời gian)

### Bước 4: Intent Resolution

- **File**: `pipeline/dialogue_manager.py` → `_resolve_intent()`
- Follow-up intents (ASK_LOCATION, ASK_PRICE, ASK_OPEN_TIME) được merge vào RECOMMEND khi đang trong flow gợi ý
- Confidence < threshold → giữ intent hiện tại
- SMALL_TALK / OUT_OF_SCOPE luôn giữ nguyên

### Bước 5: DST Update

- **File**: `dialogue_state_tracking/dst.py` → `DialogueStateTracker.update_state()`

| Sub-step | File | Class |
|----------|------|-------|
| Slot Validation | `slot_guard.py` | `SlotValidator` — confidence threshold, chuẩn hóa location/price/taste, lọc noise |
| Intent Shift Detection | `intent_shift_detector.py` | `IntentShiftDetector` — cancel, goodbye, change món, negate dish |
| Semantic Merging | `semantic_slot_ranking.py` | `SemanticSlotRanker` — specificity scoring, force replace khi shift |
| State Commit | `state_schema.py` | `DialogueState` — add_turn, cập nhật filled_slots, tính state_quality |

### Bước 6: Policy Decision

- **File**: `dialogue_policy/hybrid_policy.py` → `HybridPolicy.decide_action()`

**3 lớp quyết định**:
1. **RuleBasedPolicy** (`rule_based_policy.py` + `policy_rules.json`): 6 safety rules, priority 1-11
2. **HuggingFaceLLMPolicy** (`hf_llm_policy.py`): Qwen2.5 với structured JSON output
3. **Deterministic Fallback**: ASK_SLOT nếu thiếu slot, RECOMMEND nếu đủ

**State Quality Guards**: tự động downgrade RECOMMEND → ASK_SLOT nếu thiếu slot, upgrade ASK_SLOT → RECOMMEND nếu đã đủ

### Bước 7: Retrieval (Cải tiến)

### Bước 8: Response

- Template selection với anti-repetition (kiểm tra 3 turn gần nhất)
- Gộp recommendation list vào response nếu có

---

## Hệ thống Retrieval

### Kiến trúc Retrieval Mới (Improved Pipeline)

```
Query + Slots
    │
    ├─ 1. Encode Query ─── SentenceTransformer → query embedding vector
    │
    ├─ 2. Sparse Search ── PostgreSQL FTS (ts_rank + websearch_to_tsquery trên menus)
    │     └─ File: sparse_retriever.py → SparseRetriever.search()
    │     └─ Output: List[{menu_id, restaurant_id, dish_name, score}]
    │
    ├─ 3. Dense Search ─── pgvector (embedding <-> query_vec, MIN distance GROUP BY restaurant_id)
    │     └─ File: dense_retriever.py → DenseRetriever.search()
    │     └─ Output: List[(restaurant_id, distance)]
    │
    ├─ 4. RRF Fusion ─── Reciprocal Rank Fusion (k=60)
    │     └─ File: fusion.py → rrf_fusion()
    │     └─ Merge sparse + dense rankings thành danh sách restaurant_ids
    │
    ├─ 5. SQL Filter ─── Haversine distance + price + time filter (restaurant-level)
    │     └─ File: sql_retriever.py → SQLRetriever.filter_by_candidates()
    │     └─ Output: List[{id, name, price_level, lat, lng, distance}]
    │
    ├─ 6. CrossEncoder Rerank ─── BAAI/bge-reranker-base
    │     └─ File: reranker.py → CrossEncoderReranker.rerank()
    │     └─ Score từng (query, menu_text) pair
    │
    ├─ 7. Enrich ─── review counts, tags, menus, open_time_slot, best_dish
    │
    └─ 8. LightGBM Rerank (optional) ─── 19 features
          └─ File: ranking/ranker.py → LightGBMReranker
```

### So sánh Retrieval Cũ vs Mới

| Khía cạnh | Pipeline Cũ (Legacy) | Pipeline Mới (Improved) |
|-----------|---------------------|------------------------|
| **Level** | Restaurant-level | Menu-level |
| **Sparse** | TF-IDF in-memory | PostgreSQL FTS (ts_rank) |
| **Dense** | Cosine similarity in-memory | pgvector ANN (`<->` operator) |
| **Fusion** | Weighted sum (0.4 × TF-IDF + 0.6 × Embed) | Reciprocal Rank Fusion (RRF) |
| **Reranker** | LightGBM only | CrossEncoder + LightGBM |
| **Filter** | SQL spatial (haversine approx) | SQL spatial (haversine chính xác) |
| **Embedding Model** | paraphrase-multilingual-MiniLM-L12-v2 | paraphrase-multilingual-MiniLM-L12-v2 |
| **CrossEncoder** | Không có | BAAI/bge-reranker-base |

### Retrieval Settings

Cấu hình trong `RetrievalSettings` dataclass (`hybrid_retriever.py`):

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `use_improved_pipeline` | True | Bật/tắt pipeline cải tiến |
| `use_sparse` | True | Bật/tắt PostgreSQL FTS |
| `use_dense` | True | Bật/tắt pgvector |
| `use_cross_encoder` | True | Bật/tắt CrossEncoder reranker |
| `cross_encoder_model` | "BAAI/bge-reranker-base" | Model CrossEncoder |
| `sparse_top_k` | 50 | Số kết quả sparse |
| `dense_top_k` | 50 | Số kết quả dense |
| `rrf_k` | 60 | Hằng số RRF |
| `sparse_fallback_threshold` | 10 | Ngưỡng fallback sparse → dense global |
| `sql_filter_limit` | 20 | Số kết quả sau SQL filter |
| `top_k` | 5 | Số kết quả cuối cùng |
| `top_n` | 200 | Số candidates SQL (legacy) |
| `embedding_model_name` | "paraphrase-multilingual-MiniLM-L12-v2" | Model embedding |
| `rerank_enabled` | False | Bật/tắt LightGBM rerank |
| `rerank_model_path` | None | Đường dẫn model LightGBM |

---

## Cấu trúc Thư mục và Trạng thái Module

### Đã triển khai đầy đủ

| Module | Files | Mô tả |
|--------|-------|-------|
| `dialogue_state_tracking/` | `state_schema.py`, `dst.py`, `intent_shift_detector.py`, `semantic_slot_ranking.py`, `slot_guard.py` | DST, 12 intents, slot validation, intent shift detection, semantic merging |
| `dialogue_policy/` | `hybrid_policy.py`, `rule_based_policy.py`, `hf_llm_policy.py`, `ml_policy.py`, `policy_rules.json` | Option 3 policy: LLM + rules + ML fallback |
| `retrieval/` | `hybrid_retriever.py`, `sparse_retriever.py`, `dense_retriever.py`, `sql_retriever.py`, `vector_retriever.py`, `fusion.py`, `reranker.py`, `labeling_data.py` | Improved pipeline: FTS + pgvector + RRF + CrossEncoder |
| `pipeline/` | `dialogue_manager.py`, `domain_gate.py`, `app_gradio_hybrid.py`, `run_orchestrator_demo*.py` | Orchestrator, domain gate, Gradio UI, CLI demos |
| `ranking/` | `ranker.py` | LightGBM reranker (19 features) |
| `scripts/` | `eval_dst.py`, `train_ml_policy.py`, `build_policy_dataset.py` | Đánh giá DST, huấn luyện ML policy |

### Placeholder / Chưa triển khai

| Module | Files | Dự kiến |
|--------|-------|---------|
| `context_builder/` | `context_builder.py`, `templates/` | Xây dựng prompt context từ DST + history |
| `query_processing/` | `embedder.py`, `normalizer.py`, `query_schema.py` | Chuẩn hóa và embedding query |
| `response_generation/` | `llm_generator.py`, `post_processor.py`, `prompts/` | Sinh response bằng LLM |
| `user_memory/` | `memory_manager.py`, `preference_tracker.py` | Ghi nhớ sở thích người dùng |
| `logging/` | `logger.py`, `analytics.py` | Logger trung tâm, thống kê metrics |
| `intent_classification/` | `train.py`, `model/` | Huấn luyện intent classifier |
| `slot_extraction/` | `train.py`, `model/` | Huấn luyện slot extractor |
| `pipeline/api.py` | — | FastAPI REST endpoint |
| `pipeline/chatbot_pipeline.py` | — | Pipeline assembly |

---

## Cơ sở Dữ liệu

### PostgreSQL (Supabase)

Kết nối qua biến môi trường: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SSLMODE`

| Bảng | Cột chính | Mục đích |
|------|-----------|----------|
| `restaurants` | id, name, price_level, latitude, longitude, open_time_slot | Thông tin cơ bản nhà hàng |
| `restaurant_embeddings` | restaurant_id, content, embedding | Văn bản + vector cho semantic search (legacy) |
| `menus` | id, restaurant_id, dish_name, description, tags, embedding, tsv | Menu món ăn + pgvector embedding + FTS tsvector |
| `user_ratings` | user_id, restaurant_id, rating | Đánh giá từ người dùng |

### Dữ liệu Local

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `retrieval/database/foods.csv` | 7.3 MB | 1000+ món ăn Việt Nam từ MonNgonMoiNgay |
| `retrieval/database/ratings.csv` | 2.1 MB | Ma trận đánh giá user-food |
| `retrieval/database/evaluation_IR.csv` | 3.6 KB | 15 câu query mẫu cho IR evaluation |
| `retrieval/labels.json` | 30 KB | Nhãn relevance query-restaurant (1250+ entries) |
| `data/policy_train.jsonl` | 1.3 KB | 8 seed rows cho ML policy training |
| `data/dst_eval.jsonl` | — | Dữ liệu đánh giá DST |

---

## Công nghệ Sử dụng

| Package | Mục đích |
|---------|----------|
| `torch` | Deep learning backend |
| `transformers` | HuggingFace models: intent classifier, slot extractor, LLM policy (Qwen2.5) |
| `sentence-transformers` | Embedding model + CrossEncoder reranker |
| `scikit-learn` | TF-IDF, LogisticRegression (ML policy) |
| `lightgbm` | Reranking model (19 features) |
| `psycopg2` | PostgreSQL connection |
| `pandas`, `numpy` | Data processing |
| `gradio` | Web demo UI |
| `streamlit` | Labeling tool UI |
| `joblib` | Model serialization |
| `pytest` | Testing |

### Models

| Model | Use |
|-------|-----|
| Custom HF classifier (12 classes) | Intent classification |
| Custom HF NER (BIO tags) | Slot extraction |
| `Qwen/Qwen2.5-7B-Instruct` hoặc `3B` | LLM policy decision |
| `paraphrase-multilingual-MiniLM-L12-v2` | Query + document embedding |
| `BAAI/bge-reranker-base` | CrossEncoder reranker |

---

## Hướng dẫn Cài đặt

```bash
# 1. Clone repository
git clone <repo-url>
cd chatbot_recommend

# 2. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Cài PyTorch (tuỳ chọn CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Cài dependencies
pip install transformers sentence-transformers scikit-learn lightgbm pandas numpy joblib psycopg2-binary gradio streamlit pytest

# 5. Cấu hình database (PostgreSQL + pgvector)
# Set các biến môi trường:
#   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE
# Hoặc tạo file configs/secrets.env

# 6. Tải/copy model vào thư mục models/
#   models/intent_classifier/
#   models/slot_extractor/
```

---

## Hướng dẫn Chạy

### Gradio Web UI

```bash
python pipeline/app_gradio_hybrid.py \
    --intent-model models/intent_classifier \
    --slot-model models/slot_extractor \
    --llm-model Qwen/Qwen2.5-3B-Instruct \
    --default-lat 10.762622 \
    --default-lon 106.660172
```

Flags chính:
- `--llm-model`: model LLM policy (để trống nếu không dùng LLM)
- `--llm-device`: auto/cuda/cpu
- `--default-lat`, `--default-lon`: tọa độ mặc định
- `--default-budget`: mức giá mặc định
- `--default-distance-km`: bán kính tìm kiếm (km)
- `--retrieval-top-k`: số lượng gợi ý (default: 5)
- `--cross-encoder-model`: model CrossEncoder (default: BAAI/bge-reranker-base)
- `--no-sparse`: tắt FTS search
- `--no-dense`: tắt pgvector search
- `--no-cross-encoder`: tắt CrossEncoder reranker
- `--no-improved-pipeline`: dùng pipeline cũ (legacy)
- `--rerank-model-path`: đường dẫn LightGBM model
- `--debug`: bật debug log

### CLI Demo

```bash
# Hybrid (đầy đủ: rule + LLM policy)
python pipeline/run_orchestrator_demo_hybrid.py \
    --intent_model_path models/intent_classifier \
    --slot_model_path models/slot_extractor \
    --use_llm --llm_model Qwen/Qwen2.5-3B-Instruct

# Rule-based (cơ bản)
python pipeline/run_orchestrator_demo.py \
    --intent_model_path models/intent_classifier \
    --slot_model_path models/slot_extractor
```

### Chạy Tests

```bash
pytest tests/ -q -v
```

### Scripts

```bash
# Tạo seed dataset cho ML policy
python scripts/build_policy_dataset.py --output data/policy_train.jsonl

# Huấn luyện ML policy (LogisticRegression + TF-IDF)
python scripts/train_ml_policy.py --train_jsonl data/policy_train.jsonl --output models/ml_policy.joblib

# Đánh giá DST
python scripts/eval_dst.py \
    --intent_model_path models/intent_classifier \
    --slot_model_path models/slot_extractor \
    --eval_jsonl data/dst_eval.jsonl
```

### Streamlit Labeling Tool

```bash
streamlit run retrieval/labeling_data.py
```

Công cụ đánh giá relevance query-restaurant: nhập query → load candidates (semantic similarity + random sampling) → đánh giá 0-3 → lưu vào `labels.json`.

---

## Hệ thống Policy

### 3 Lớp Quyết định

1. **RuleBasedPolicy** (`dialogue_policy/rule_based_policy.py` + `policy_rules.json`)
   - 6 safety rules, sắp xếp theo priority (1-11)
   - Chỉ active như safety guard: SMALL_TALK, OUT_OF_SCOPE, NO_CLEAR_INTENT
   - Rules: cancel/goodbye, ask_missing_location, clarify_vague_location, recommend_when_complete, handle_small_talk, handle_out_of_scope

2. **HuggingFaceLLMPolicy** (`dialogue_policy/hf_llm_policy.py`)
   - Sử dụng Qwen2.5 (3B/7B) với structured JSON output prompt
   - Action space: ASK_SLOT, CLARIFY, CONFIRM, RECOMMEND, RESPOND, FALLBACK
   - Prompt constraints: không RECOMMEND nếu thiếu slot, ưu tiên slot đang thiếu

3. **SklearnMLPolicy / Deterministic Fallback** (`dialogue_policy/ml_policy.py`)
   - TF-IDF + LogisticRegression, train từ `data/policy_train.jsonl`
   - Fallback: ASK_SLOT nếu thiếu slot, RECOMMEND nếu đủ

### State Quality Guards

- `get_state_quality()`: 1.0 - 0.12×low_conf - 0.08×unconfirmed - 0.10×conflicts
- Threshold: 0.65 (mặc định)
- RECOMMEND + missing slots → downgrade ASK_SLOT
- ASK_SLOT + complete → upgrade RECOMMEND
- block_recommend flag → downgrade CLARIFY

---

## Cơ chế DST (Dialogue State Tracking)

### State Schema

- **IntentType enum**: 12 intents
- **Slot dataclass**: type, value, confidence, turn_index, source_intent, history
- **Turn dataclass**: turn_index, user_utterance, intent, slots, bot_response, bot_action
- **DialogueState dataclass**: session_id, turns, filled_slots, current_intent, context, slot_conflicts

### Intent Shift Detection

| Shift type | Trigger | Action |
|------------|---------|--------|
| CANCEL | "không muốn ăn nữa", "dừng lại", "bỏ qua", "hủy" | Hard reset state, block_recommend |
| GOODBYE | "tạm biệt", "bye", "hẹn gặp lại" | Soft reset, block_recommend |
| CHANGE | "đổi món", "món khác", "ăn món khác" | Giữ intent, replace slot cũ |
| NEGATE | "không muốn ăn [món]" | Drop DISH slot, block_recommend |
| FOLLOWUP_LOCATION | "quận 1" trong RECOMMEND flow | Force replace LOCATION slot |

---

## Hướng dẫn Phát triển

### Thêm Intent mới
1. Thêm vào `IntentType` enum trong `state_schema.py`
2. Thêm `required_slots_map` entry trong `get_required_slots()`
3. Cập nhật model intent classifier (re-train)
4. Cập nhật `LABEL_ID_TO_INTENT` mapping trong `dialogue_manager.py`
5. Thêm DomainGate handling nếu cần

### Thêm Policy Rule mới
1. Thêm rule vào `policy_rules.json` (theo format JSON, hỗ trợ comment `//`)
2. Đặt priority hợp lý (cao hơn = ưu tiên hơn)
3. Định nghĩa condition (intent, missing_slots, turn_count, slots, state_complete)
4. Định nghĩa action (type, slot_to_ask, templates)

### Mở rộng Retrieval
- Thêm feature vào `build_features()` trong `hybrid_retriever.py`
- Thêm enrichment function (pattern: `_get_X_batch`)
- Train LightGBM model → set `rerank_enabled=True, rerank_model_path=...`
- Điều chỉnh `sparse_top_k`, `dense_top_k`, `rrf_k` trong `RetrievalSettings`

---

## Kiểm thử

### Trạng thái test hiện tại

| File test | Số tests | Trạng thái |
|-----------|----------|------------|
| `tests/test_intent_shift_detector.py` | 4 | Passing |
| `tests/test_hybrid_policy.py` | 8 | Passing |
| `tests/test_dst_policy.py` | 2 | Passing |
| `tests/test_pipeline_integration.py` | 4 | Cần `conftest.py` với `rules_path` fixture |
| `tests/test_dst.py` | 0 | Empty stub |
| `tests/test_end_to_end.py` | 0 | Empty stub |
| `tests/test_policy.py` | 0 | Empty stub |

### Known Issue

`test_pipeline_integration.py` cần file `tests/conftest.py` với:

```python
import pytest
from pathlib import Path

@pytest.fixture
def rules_path():
    return str(Path(__file__).resolve().parent.parent / "dialogue_policy" / "policy_rules.json")
```

---

## File Quan trọng

| File | Vai trò |
|------|---------|
| `pipeline/dialogue_manager.py` | Main orchestrator + Intent/Slot classifier |
| `dialogue_state_tracking/state_schema.py` | Data classes cho toàn bộ state |
| `dialogue_state_tracking/dst.py` | Dialogue state tracker core |
| `dialogue_policy/hybrid_policy.py` | Option 3 policy decision engine |
| `dialogue_policy/rule_based_policy.py` | Rule-based safety layer |
| `dialogue_policy/hf_llm_policy.py` | LLM wrapper cho policy |
| `dialogue_policy/policy_rules.json` | 6 safety rules |
| `retrieval/hybrid_retriever.py` | Improved retrieval pipeline (FTS + pgvector + RRF + CrossEncoder) |
| `retrieval/sparse_retriever.py` | PostgreSQL FTS search |
| `retrieval/dense_retriever.py` | pgvector dense search |
| `retrieval/fusion.py` | RRF fusion algorithm |
| `retrieval/reranker.py` | CrossEncoder reranker |
| `retrieval/sql_retriever.py` | SQL retriever + filter_by_candidates |
| `pipeline/app_gradio_hybrid.py` | Gradio web UI |
| `pipeline/domain_gate.py` | Regex-based domain filtering |
| `dialogue_state_tracking/intent_shift_detector.py` | Intent shift/cancel/change detection |
| `dialogue_state_tracking/slot_guard.py` | Slot validation + normalization |

---

## Lưu ý Vận hành

- `configs/secrets.env` để lưu secrets. Thêm vào `.gitignore`.
- `configs/config.yaml` hiện tại trống — cấu hình được truyền qua CLI args.
- Model intent classifier và slot extractor cần được train riêng và đặt trong `models/`.
- LLM policy cần GPU >= 16GB (Qwen2.5-7B) hoặc >= 8GB (Qwen2.5-3B).
- Improved retrieval pipeline yêu cầu PostgreSQL có extension `pgvector` và cột `tsv` (tsvector) trong bảng `menus`.
- Để dùng pipeline cũ (legacy), thêm flag `--no-improved-pipeline`.
