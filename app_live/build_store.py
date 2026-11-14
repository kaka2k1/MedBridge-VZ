"""Dựng kho văn bản SQLite (lang, doc_id) -> url, title, text cho app tra cứu trực tiếp."""
import glob, sqlite3, pandas as pd
R = "/home/phong/Desktop/hsk/test2/"
db = sqlite3.connect(R + "app_live/docs.sqlite"); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
db.execute("CREATE TABLE IF NOT EXISTS d(lang TEXT, id INTEGER, url TEXT, title TEXT, text TEXT, PRIMARY KEY(lang,id))")
n = 0
for fn in sorted(glob.glob(R + "data/docs/corpus-*.parquet")):
    df = pd.read_parquet(fn, columns=["doc_id", "url", "title", "text"])
    db.executemany("INSERT OR REPLACE INTO d VALUES('vi',?,?,?,?)", ((int(a), u, t, x) for a, u, t, x in zip(df.doc_id, df.url, df.title, df.text))); db.commit(); n += len(df); print(fn[-20:], n, flush=True)
df = pd.read_parquet(R + "data/cache/kg_buildin_d/zh_docs.parquet", columns=["doc_id", "url", "title", "text"])
db.executemany("INSERT OR REPLACE INTO d VALUES('zh',?,?,?,?)", ((int(a), u, t, x) for a, u, t, x in zip(df.doc_id, df.url, df.title, df.text))); db.commit()
print("zh", len(df)); print("total", db.execute("select lang,count(*) from d group by lang").fetchall())
