
# Chatbot Recommend (Option 3)

Du an chatbot goi y quan an voi kien truc Option 3: LLM la nguoi ra quyet dinh, Rule la lop an toan, DST la bo nho hoi thoai.
README nay mo ta chi tiet muc dich va y nghia tung thu muc va tung file trong du an.

## Tong quan

- Muc tieu: Go y quan an theo mon, khu vuc, gia ca, va so thich nguoi dung.
- Pipeline chinh: Intent + Slot -> DST -> Hybrid Policy -> Response.
- Che do ho tro: Gradio demo, API, script huan luyen/eval, bo ghi log, va cac bo thu.

## Kien truc tong the (rut gon)

1) **Intent + Slot**: Nhan y dinh va trich xuat slot tu cau hoi.
2) **DST**: Cap nhat trang thai (filled_slots, missing_slots, chat context).
3) **Hybrid Policy (Option 3)**: LLM ra quyet dinh, Rule chi la guardrail an toan.
4) **Retrieval + Ranking + Response**: (tuy chon) truy van du lieu, xep hang, va sinh cau tra loi.

## Huong dan nhanh

- Chay Gradio demo:
	- `python pipeline/app_gradio_hybrid.py`
- Chay demo orchestrator:
	- `python pipeline/run_orchestrator_demo_hybrid.py`
	- `python pipeline/run_orchestrator_demo.py`
- Chay test:
	- `pytest tests -q`

Ghi chu: Cau hinh model va duong dan model xem trong `configs/config.yaml`.

---

## Cau truc thu muc va y nghia tung file

### Root

- `README.md`: Tai lieu mo ta toan bo du an (file nay).

### configs/

- `config.yaml`: Cau hinh chung (duong dan model, tham so runtime, nguong, v.v.).
- `secrets.env`: Bien moi truong/secret (API key, thong tin ket noi). **Khong commit len git** neu co du lieu nhay cam.

### context_builder/

- `context_builder.py`: Xay dung context/prompt tu DST va lich su hoi thoai.
- `templates/`: Thu muc chua template prompt (hien dang trong).

### data/

- `dst_eval.jsonl`: Du lieu danh gia DST (dung cho test/benchmark).
- `policy_train.jsonl`: Du lieu huan luyen policy (state -> action).

### dialogue_policy/

- `actions.py`: Dinh nghia hanh dong/policy helpers (action types, mapping, tien ich).
- `hf_llm_policy.py`: Wrapper LLM (HuggingFace) de ra quyet dinh action theo schema.
- `hybrid_policy.py`: Chinh sach Option 3 (LLM + guardrail + DST).
- `ml_policy.py`: Policy ML (Sklearn) dung cho train/thu nghiem (runtime co the tat).
- `policy_rules.json`: Tap rule an toan (small talk, out of scope, v.v.).
- `rule_based_policy.py`: Policy rule-based engine de thuc thi rules.

### dialogue_state_tracking/

- `dst.py`: Dialogue State Tracker (update state, merge slots, log conflicts).
- `intent_shift_detector.py`: Phat hien intent shift/cancel/goodbye/change.
- `semantic_slot_ranking.py`: Quy tac chon slot moi vs cu (semantic ranking).
- `slot_guard.py`: Validate + normalize slot (filter noise, patterns, thresholds).
- `state_schema.py`: Schema cho intent/turn/slot/state.
- `tests/`: Thu muc test rieng cho DST (hien dang trong).

### intent_classification/

- `train.py`: Script huan luyen intent classifier.
- `model/`: Thu muc chua model da huan luyen (hien dang trong).

### logging/

- `analytics.py`: Tong hop thong ke/metrics (latency, quality, v.v.).
- `logger.py`: Cau hinh logger va helper ghi log.

### pipeline/

- `api.py`: Mo/fast API gateway cho chatbot (neu su dung REST).
- `app_gradio_hybrid.py`: UI Gradio cho debug/present.
- `chatbot_pipeline.py`: Pipeline ket hop intent/slot/DST/policy.
- `dialogue_manager.py`: Orchestrator quan ly luong hoi thoai.
- `domain_gate.py`: Chan out-of-domain, small talk, goodbye/cancel.
- `run_orchestrator_demo.py`: Demo pipeline co ban.
- `run_orchestrator_demo_hybrid.py`: Demo pipeline hybrid (Option 3).

### query_processing/

- `embedder.py`: Tao embedding cho query.
- `normalizer.py`: Chuan hoa query (lower, remove noise, v.v.).
- `query_schema.py`: Dinh nghia schema cho query.

### ranking/

- `ranker.py`: Xep hang ket qua tu retrieval.
- `scoring_functions.py`: Ham tinh diem (score) cho ranking.

### response_generation/

- `llm_generator.py`: Sinh response (LLM or template).
- `post_processor.py`: Hau xu ly response (format, sanitize, v.v.).
- `prompts/`: Thu muc prompt templates (hien dang trong).

### retrieval/

- `hybrid_retriever.py`: Retriever ket hop (vector + sql).
- `sql_retriever.py`: Retriever truy van DB.
- `vector_retriever.py`: Retriever vector search.
- `database/`
	- `scheme.sql`: Schema DB (tao bang, index).
	- `seed_data.py`: Seed du lieu mau.

### scripts/

- `build_policy_dataset.py`: Tao dataset policy tu log/state.
- `eval_dst.py`: Danh gia DST (metrics).
- `train_ml_policy.py`: Huan luyen ML policy (Sklearn).

### slot_extraction/

- `train.py`: Script huan luyen slot extractor.
- `model/`: Thu muc chua model da huan luyen (hien dang trong).

### tests/

- `test_dst.py`: Unit test cho DST co ban.
- `test_dst_policy.py`: Test DST + policy integration.
- `test_end_to_end.py`: Test tu dau den cuoi.
- `test_hybrid_policy.py`: Test cho HybridPolicy (Option 3).
- `test_intent_shift_detector.py`: Test cho intent shift/cancel/change.
- `test_pipeline_integration.py`: Test pipeline tich hop.
- `test_policy.py`: Test cho policy (rule/logic).

### user_memory/

- `memory_manager.py`: Quan ly bo nho nguoi dung.
- `preference_tracker.py`: Luu va cap nhat so thich nguoi dung.

---

## Luu y van hanh

- `secrets.env` chi de luu secrets. Hay them vao `.gitignore` neu chua co.
- Thu muc `model/` trong `intent_classification/` va `slot_extraction/` la noi luu model sau huan luyen.
- Voi `Option 3`, LLM la nguon quyet dinh chinh, Rule chi la guardrail.

## Dong gop

- Neu muon bo sung tinh nang, hay cap nhat test tuong ung trong `tests/`.
- Neu thay doi schema slot/intent, can cap nhat `state_schema.py` va test lien quan.
