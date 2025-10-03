import os, sqlite3, numpy as np, faiss, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

APPDATA = os.environ.get("APPDATA", str(Path.home() / ".model_finder"))
DB_DIR = Path(APPDATA) / "ModelFinder"
DB_PATH = DB_DIR / "index.db"
ART_DIR = Path("db"); ART_DIR.mkdir(parents=True, exist_ok=True)
IDX_PATH = ART_DIR / "faiss_tfidf.index"
VEC_PATH = ART_DIR / "tfidf_vectorizer.joblib"
IDS_PATH = ART_DIR / "faiss_ids.npy"

def load_rows():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, path, name, ext, tags FROM files ORDER BY id ASC")
    rows = cur.fetchall()
    con.close()
    return rows

def build_text(path: str, name: str, ext: str, tags: str) -> str:
    p = Path(path or "")
    parents = []
    if p.parent.name: parents.append(p.parent.name)
    if p.parent.parent and p.parent.parent.name: parents.append(p.parent.parent.name)
    parts = [
        Path(name or "").stem,
        (ext or "").lstrip("."),
        tags or "",
        *parents
    ]
    return " ".join(x for x in parts if x).strip()

def main():
    rows = load_rows()
    if not rows:
        print("No rows in DB. Run the GUI and scan folders first.")
        return

    texts, ids = [], []
    for _id, path, name, ext, tags in rows:
        t = build_text(path, name, ext, tags)
        if t:
            texts.append(t)
            ids.append(_id)

    if not texts:
        print("All rows produced empty text. Add tags or verify filenames, then retry.")
        return

    vec = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w\-\.\#/]+\b",
        ngram_range=(1, 2),
        min_df=2,          # drops singletons/noise
        max_features=75000 # a bit more vocabulary headroom
    )
    X = vec.fit_transform(texts).astype("float32")
    X = normalize(X, norm="l2", copy=False)
    X = X.toarray().astype("float32")  # OK for small/med corpora

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(IDX_PATH))
    joblib.dump(vec, VEC_PATH)
    np.save(IDS_PATH, np.asarray(ids, dtype=np.int64))

    print(f"Indexed {len(ids)} items.")
    print(f"- {IDX_PATH}\n- {VEC_PATH}\n- {IDS_PATH}")

if __name__ == "__main__":
    main()
