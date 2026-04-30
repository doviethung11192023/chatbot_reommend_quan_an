import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# CONFIG
# ======================
DB_CONFIG = {
    "host": "aws-1-ap-northeast-2.pooler.supabase.com",
    "port": 6543,
    "database": "postgres",
    "user": "postgres.ypkwqsbsunlvpoqdadbq",
    "password": "5$eAK8EV4S+gsKj",
    "sslmode": "require"
}

POOL_SIZE = 25  # số quán để label mỗi query

# ======================
# CONNECT DB
# ======================
@st.cache_resource
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

conn = get_conn()

# ======================
# LOAD MODEL (MiniLM chỉ dùng để build pool, KHÔNG dùng để label)
# ======================
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# ======================
# LOAD ALL RESTAURANTS
# ======================
def get_all_restaurants():
    query = """
    SELECT r.id, r.name, r.price_level,
           COALESCE(AVG(ur.rating),0) as rating,
           re.embedding
    FROM restaurants r
    JOIN restaurant_embeddings re ON r.id = re.restaurant_id
    LEFT JOIN user_ratings ur ON r.id = ur.restaurant_id
    GROUP BY r.id, re.embedding
    """

    df = pd.read_sql(query, conn)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))
    return df

# ======================
# LOAD MENU
# ======================
def get_menu(restaurant_id):
    query = """
    SELECT dish_name, description, tags
    FROM menus
    WHERE restaurant_id = %s
    """
    return pd.read_sql(query, conn, params=(restaurant_id,))

# ======================
# BUILD CANDIDATE POOL
# ======================
def build_candidate_pool(query_text, df, pool_size=POOL_SIZE):

    query_emb = model.encode(query_text)

    # Semantic similarity (MiniLM)
    df["sim"] = df["embedding"].apply(
        lambda x: cosine_similarity([query_emb], [x])[0][0]
    )

    top_semantic = df.sort_values("sim", ascending=False).head(12)

    # Random hard negatives
    random_sample = df.sample(15)

    # Combine
    pool = pd.concat([top_semantic, random_sample]).drop_duplicates(subset=["id"])

    return pool.head(pool_size)

# ======================
# UI
# ======================
st.title("🍜 Restaurant Labeling Tool (IR Benchmark)")

query = st.text_input("🔍 Nhập query (VD: 'ăn cay miền trung', 'trà sữa topping')")

if st.button("🚀 Load Candidates"):

    if not query.strip():
        st.warning("⚠️ Nhập query trước")
    else:
        full_df = get_all_restaurants()
        pool_df = build_candidate_pool(query, full_df)

        st.session_state["candidates"] = pool_df
        st.session_state["labels"] = {}

# ======================
# DISPLAY CANDIDATES
# ======================
if "candidates" in st.session_state:

    df = st.session_state["candidates"]

    st.info(f"👉 Label {len(df)} restaurants cho query này")

    for i, row in df.iterrows():

        st.markdown("---")
        st.subheader(f"{row['name']}")
        st.write(f"⭐ Rating: {row['rating']:.1f} | 💰 Price: {row['price_level']}")

        menu_df = get_menu(row["id"])

        with st.expander("🍽 Menu"):
            for _, dish in menu_df.iterrows():
                st.write(f"- {dish['dish_name']}")
                if dish["description"]:
                    st.caption(dish["description"])
                if dish["tags"]:
                    st.caption(f"🏷 {dish['tags']}")

        # 👉 Label manual (KHÔNG auto suggest)
        score = st.select_slider(
            f"Relevance score",
            options=[0,1,2,3],
            value=0,
            key=f"slider_{i}"
        )

        st.session_state["labels"][int(row["id"])] = int(score)

# ======================
# SAVE LABELS
# ======================
if st.button("💾 Save Labels"):

    if "labels" not in st.session_state:
        st.warning("⚠️ Chưa có dữ liệu")
    else:

        data = []

        for rid, score in st.session_state["labels"].items():
            data.append({
                "query": query,
                "restaurant_id": rid,
                "relevance": score
            })

        # Load existing file
        try:
            with open("labels.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except:
            existing_data = []

        existing_data.extend(data)

        with open("labels.json", "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        st.success("✅ Saved thành công!")