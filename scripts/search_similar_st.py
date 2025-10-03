# scripts/search_similar_st.py
import os, sqlite3, numpy as np, faiss, argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

APPDATA = os.environ.get("APPDATA", str(Path.home() / ".model_finder"))
DB_DIR = Path(APPDATA) / "ModelFinder"
DB_PATH = DB_DIR / "index.db"
ART_DIR = Path("db")
IDX_PATH = ART_DIR / "faiss_st.index"
IDS_PATH = ART_DIR / "faiss_ids.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
    g.add_argument("--like-id", type=int)
    g.add_argument("--text", type=str)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index(str(IDX_PATH))
    ids = np.load(IDS_PATH)

    if args.like_id:
        qtext = get_text_for_id(args.like_id)
        if not qtext: print("ID not found"); return
    else:
        qtext = args.text

    q = model.encode([qtext], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, args.topk)
    # Print like before...
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        aid = int(ids[idx])
        cur.execute("SELECT path, name, ext, tags FROM files WHERE id=?", (aid,))
        row = cur.fetchone()
        if row:
            path, name, ext, tags = row
            # replace inside the pretty print loop
            display = name
            if not name.lower().endswith(ext.lower()):
                display = f"{name}{ext}"
            print(f"{rank:>2}. score={score:.3f}  id={aid}  {display}  [{tags}]")
            print(f"    {path}")
    con.close()

if __name__ == "__main__":
    main()
