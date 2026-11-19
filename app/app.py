import os
import streamlit as st
import sys
import re
import html
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.bm25 import load_index as bm25_load, bm25_search
from src.semantic import load_faiss_index, semantic_search
from src.hybrid import load_indexes
from src.rag_pipeline import HybridRAGPipeline
from src.artifacts import artifacts_ready, ensure_artifacts

data_dir = project_root / "data" / "processed"

st.set_page_config(
    page_title="Musical Instruments & Equipment Search",
    page_icon="🎸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling: dark, high-contrast theme. Colours mirror .streamlit/config.toml so
# the page looks right even if Streamlit is launched from another directory.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0B0F14;
        --surface: #161C24;
        --surface-2: #1E2630;
        --border: #2E3846;
        --text: #F5F7FA;
        --muted: #B8C2CF;
        --accent: #FFB000;
        --accent-2: #22D3EE;
        --success: #3DDC84;
    }
    .stApp { background: var(--bg); color: var(--text); }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Hero */
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid var(--border);
        border-left: 6px solid var(--accent);
        border-radius: 12px;
        background: linear-gradient(135deg, #161C24 0%, #0F141B 100%);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: var(--text); font-size: 2rem; margin: 0 0 .35rem 0; font-weight: 800; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); margin: 0; font-size: 1rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: var(--surface); border: 1px solid var(--border);
        border-bottom: none; border-radius: 8px 8px 0 0;
        padding: .55rem 1.2rem; color: var(--muted); font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: var(--bg) !important; background: var(--accent) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* Inputs */
    .stTextInput input {
        background: var(--surface-2); color: var(--text);
        border: 1px solid var(--border); border-radius: 8px; font-size: 1.05rem;
    }
    .stTextInput input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(255,176,0,.35); }
    label, .stRadio label p { color: var(--text) !important; font-weight: 600; }

    /* Product cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        transition: border-color .15s ease;
    }
    .card:hover { border-color: var(--accent); }
    .card.ranked { border-left: 5px solid var(--success); }
    .card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
    .card-title { color: var(--text); font-size: 1.08rem; font-weight: 700; line-height: 1.35; margin: 0; }
    .card-title a { color: var(--accent-2); text-decoration: none; }
    .card-title a:hover { text-decoration: underline; }
    .card-review { color: var(--muted); font-size: .93rem; line-height: 1.5; margin: .6rem 0 .8rem 0; }
    .meta { display: flex; flex-wrap: wrap; gap: .5rem; }
    .chip {
        display: inline-block; padding: .2rem .6rem; border-radius: 999px;
        font-size: .8rem; font-weight: 700; border: 1px solid var(--border);
        background: var(--surface-2); color: var(--text); white-space: nowrap;
    }
    .chip.rating { color: var(--accent); border-color: rgba(255,176,0,.5); }
    .chip.score  { color: var(--accent-2); border-color: rgba(34,211,238,.5); }
    .chip.rank   { color: #0B0F14; background: var(--success); border-color: var(--success); }
    .rank-num {
        flex-shrink: 0; width: 2.2rem; height: 2.2rem; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        background: var(--surface-2); border: 1px solid var(--border);
        color: var(--accent); font-weight: 800;
    }

    /* RAG answer */
    .answer {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 5px solid var(--accent-2);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: var(--text);
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    .answer-label {
        color: var(--accent-2); font-size: .78rem; font-weight: 800;
        letter-spacing: .08em; text-transform: uppercase; margin-bottom: .5rem;
    }
    .section-label {
        color: var(--accent); font-size: .85rem; font-weight: 800;
        letter-spacing: .08em; text-transform: uppercase; margin: 1.2rem 0 .7rem 0;
    }
    .empty { color: var(--muted); padding: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def build_artifacts():
    """Build missing indexes once per server process (first boot on the cloud)."""
    ensure_artifacts(data_dir)
    return True


@st.cache_resource
def load_bm25():
    return bm25_load(data_dir)


@st.cache_resource
def load_semantic():
    index_path = data_dir / "faiss_index"
    return load_faiss_index(index_path, data_dir)


@st.cache_resource
def load_hybrid():
    index_path = data_dir / "faiss_index"
    return load_indexes(data_dir, index_path)


def truncate_review(reviews, limit=200):
    text = reviews if isinstance(reviews, str) else str(reviews)
    return text[:limit] + ("..." if len(text) > limit else "")


def product_card(product, position, score=None, llm_rank=None):
    """Render a single product as an HTML card."""
    title = html.escape(str(product.get("title", "")) or "Untitled product")
    asin = html.escape(str(product.get("parent_asin", "")))
    rating = html.escape(str(product.get("average_rating", "N/A")))
    review = html.escape(truncate_review(product.get("reviews", "")))

    if asin:
        title_html = f'<a href="http://www.amazon.com/dp/{asin}/ref=nosim" target="_blank">{title}</a>'
    else:
        title_html = title

    chips = [f'<span class="chip rating">★ {rating} / 5</span>']
    if score is not None:
        chips.append(f'<span class="chip score">Score {score:.4f}</span>')
    if asin:
        chips.append(f'<span class="chip">ASIN {asin}</span>')
    if llm_rank is not None:
        chips.insert(0, f'<span class="chip rank">LLM pick #{llm_rank}</span>')

    ranked_cls = " ranked" if llm_rank is not None else ""
    st.markdown(
        f"""
        <div class="card{ranked_cls}">
          <div class="card-head">
            <p class="card-title">{title_html}</p>
            <div class="rank-num">{position}</div>
          </div>
          <p class="card-review">{review or "<em>No review text available.</em>"}</p>
          <div class="meta">{"".join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
      <h1>Musical Instruments <span>&amp;</span> Equipment</h1>
      <p>Search Amazon instruments and music gear with BM25 or semantic retrieval, or ask a
      question and let the hybrid RAG pipeline recommend the best options.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ready(data_dir):
    with st.spinner("First start: building search indexes. This can take several minutes..."):
        build_artifacts()

with st.spinner("Loading search indexes..."):
    bm25_index, products_bm25 = load_bm25()
    vector_store, products_sem = load_semantic()
    hybrid_retriever = load_hybrid()
    rag_pipeline = HybridRAGPipeline(hybrid_retriever)

search, rag = st.tabs(["🔎  Search", "✨  Ask (RAG)"])

with search:
    col_query, col_method = st.columns([3, 1])
    with col_method:
        method = st.radio("Search method", ["BM25", "Semantic"], horizontal=True)
    with col_query:
        query = st.text_input("Query", placeholder="e.g. acoustic guitar strings")

    if query:
        if method == "BM25":
            results = bm25_search(query, bm25_index, products_bm25, top_k=3)
        else:
            results = semantic_search(query, vector_store, products_sem, k=3)

        st.markdown(f'<div class="section-label">Top {len(results)} results · {method}</div>',
                    unsafe_allow_html=True)
        if not results:
            st.markdown('<div class="empty">No matching products found.</div>', unsafe_allow_html=True)
        for i, (product, score) in enumerate(results, start=1):
            product_card(product, i, score=score)

with rag:
    query = st.text_input("Ask a question",
                          placeholder="e.g. What's a good beginner digital piano with weighted keys?",
                          key="rag_query")
    if query and not os.environ.get("GROQ_API_KEY"):
        # rag_pipeline.query() calls exit(1) without a key, which would kill the app.
        st.error("GROQ_API_KEY is not set. Add it as an environment variable (or in .env locally) to use RAG.")
    elif query:
        with st.spinner("Thinking..."):
            results, docs = rag_pipeline.query(query, k=5)

        if not results:
            st.error("LLM API call failed. Please check your API key or internet connection and try again later.")
        else:
            parts = re.split(r'<rank>(.*?)</rank>', results)
            if len(parts) >= 3:
                answer = parts[2]
                ranks = [r.strip() for r in parts[1].split(",")]
            else:
                answer = results
                ranks = []

            st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
            # Render with Streamlit markdown so the LLM's formatting is preserved.
            try:
                answer_box = st.container(border=True)  # Streamlit >= 1.29
            except TypeError:
                answer_box = st.container()
            with answer_box:
                st.markdown(answer.strip())

            st.markdown('<div class="section-label">Top product options</div>', unsafe_allow_html=True)
            for i, idx in enumerate(docs, start=1):
                product = hybrid_retriever.products[idx]
                asin = product.get("parent_asin", "")
                llm_rank = ranks.index(asin) + 1 if asin in ranks else None
                product_card(product, i, llm_rank=llm_rank)
