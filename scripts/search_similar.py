# scripts/search_similar.py
import os, sqlite3, numpy as np, faiss, joblib, argparse
from pathlib import Path
from sklearn.preprocessing import normalize

APPDATA = os.environ.get("APPDATA", str(Path.home() / ".model_finder"))
DB_DIR = Path(APPDATA) / "ModelFinder"
DB_PATH = DB_DIR / "index.db"
ART_DIR = Path("db")
IDX_PATH = ART_DIR / "faiss_tfidf.index"
VEC_PATH = ART_DIR / "tfidf_vectorizer.joblib"
IDS_PATH = ART_DIR / "faiss_ids.npy"

def get_text_for_id(asset_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT name, ext, tags FROM files WHERE id=?", (asset_id,))
    row = cur.fetchone()
    con.close()
    if not row: return None
    name, ext, tags = row
    return f"{name} {ext} {tags}".strip()

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--like-id", type=int, help="Asset ID to query by")
    g.add_argument("--text", type=str, help="Free-text query")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    vec = joblib.load(VEC_PATH)
    index = faiss.read_index(str(IDX_PATH))
    ids = np.load(IDS_PATH)

    if args.like_id is not None:
        qtext = get_text_for_id(args.like_id)
        if not qtext:
            print("ID not found."); return
    else:
        qtext = args.text

    q = vec.transform([qtext]).astype("float32")
    q = normalize(q, norm="l2", copy=False).toarray().astype("float32")
    D, I = index.search(q, args.topk)
    hits = [(int(ids[i]), float(D[0, j])) for j, i in enumerate(I[0])]

    # Pretty print results with names/paths
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    for rank, (aid, score) in enumerate(hits, 1):
        cur.execute("SELECT path, name, ext, tags FROM files WHERE id=?", (aid,))
        row = cur.fetchone()
        if row:
            path, name, ext, tags = row
            print(f"{rank:>2}. score={score:.3f}  id={aid}  {name}{ext}  [{tags}]")
            print(f"    {path}")
    con.close()

if __name__ == "__main__":
    main()
