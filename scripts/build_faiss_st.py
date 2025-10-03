# scripts/build_faiss_st.py
import os, sqlite3, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

APPDATA = os.environ.get("APPDATA", str(Path.home() / ".model_finder"))
DB_DIR = Path(APPDATA) / "ModelFinder"
DB_PATH = DB_DIR / "index.db"
ART_DIR = Path("db"); ART_DIR.mkdir(parents=True, exist_ok=True)
IDX_PATH = ART_DIR / "faiss_st.index"
IDS_PATH = ART_DIR / "faiss_ids.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_rows():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, name, ext, tags FROM files ORDER BY id ASC")
    rows = cur.fetchall()
    con.close()
    return rows

def main():
    rows = load_rows()
    if not rows:
        print("No rows in DB. Scan first."); return
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [f"{name} {ext} {tags}".strip() for _, name, ext, tags in rows]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    X = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X = X.astype("float32")
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(IDX_PATH))
    np.save(IDS_PATH, ids)
    print(f"Indexed {len(ids)} items with {MODEL_NAME}. Saved {IDX_PATH}")

if __name__ == "__main__":
    main()
