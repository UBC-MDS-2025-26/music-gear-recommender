"""Build the search indexes at startup if they aren't already on disk.

The generated artifacts (products.pkl, bm25_index.pkl, faiss_index/, ...) are
not committed to git, so a fresh cloud deployment (e.g. Posit Connect Cloud)
starts without them. ensure_artifacts() rebuilds whatever is missing from the
filtered parquet files that *are* committed, using the same functions as
utils2.py, so the app can boot from a clean checkout.

Environment variables:
    MAX_PRODUCTS: Number of products to index (default 10000, which matches the
        locally built indexes). Lower it for faster cold starts on small hosts.
"""
import os
from pathlib import Path

DEFAULT_MAX_PRODUCTS = 10000


def _faiss_ready(data_dir):
    index_dir = Path(data_dir) / "faiss_index"
    return (
        (Path(data_dir) / "products.pkl").exists()
        and (index_dir / "index.faiss").exists()
        and (index_dir / "index.pkl").exists()
    )


def _bm25_ready(data_dir):
    return (Path(data_dir) / "bm25_index.pkl").exists()


def artifacts_ready(data_dir):
    """Return True if every file the app loads is already on disk."""
    return _faiss_ready(data_dir) and _bm25_ready(data_dir)


def ensure_artifacts(data_dir, max_products=None, log=print):
    """Build any missing indexes from the committed parquet files.

    Args:
        data_dir: Directory holding filtered_meta.parquet / filtered_reviews.parquet,
            and where the built artifacts are written.
        max_products: Number of products to index. Defaults to the MAX_PRODUCTS
            environment variable, or 10000.
        log: Callable used to report progress (print, or a Streamlit writer).
    """
    data_dir = Path(data_dir)
    if artifacts_ready(data_dir):
        return

    # Imported lazily: utils2 pulls in torch / sentence-transformers.
    from utils2 import build_corpus_index, build_bm25_index

    if max_products is None:
        max_products = int(os.environ.get("MAX_PRODUCTS", DEFAULT_MAX_PRODUCTS))

    meta_path = data_dir / "filtered_meta.parquet"
    review_path = data_dir / "filtered_reviews.parquet"
    for path in (meta_path, review_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot build indexes: {path} is missing.")

    if not _faiss_ready(data_dir):
        log(f"Building product corpus and FAISS index ({max_products} products)...")
        build_corpus_index(meta_path, review_path, data_dir, max_products=max_products)
        # BM25 is built from products.pkl, so it must be rebuilt to stay in sync.
        (data_dir / "bm25_index.pkl").unlink(missing_ok=True)

    if not _bm25_ready(data_dir):
        log("Building BM25 index...")
        build_bm25_index(data_dir)

    log("Indexes ready.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ensure_artifacts(project_root / "data" / "processed")
