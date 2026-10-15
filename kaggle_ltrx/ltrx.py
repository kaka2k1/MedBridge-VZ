"""Kaggle CPU: chạy scripts/ltr.py trên đặc trưng đã dựng sẵn (x.pkl = --xcache của cấu hình v57) + đặc trưng --xq (doc là gold của câu tương tự khác).
Dataset r2ai-ltrx: x.pkl, details_all.jsonl (nhãn rút gọn), e5_queries.npy/e5_qids.json, ltr.py, rerank_vi.py. Output: ltr_order_oof28*.json, log."""
import glob, os, shutil, subprocess, sys
W = "/kaggle/working"; I = os.path.dirname(glob.glob("/kaggle/input/**/x.pkl", recursive=True)[0])
os.makedirs(f"{W}/scripts", exist_ok=True); os.makedirs(f"{W}/data/eval", exist_ok=True); os.makedirs(f"{W}/e5", exist_ok=True)
for f in ("ltr.py", "rerank_vi.py"): shutil.copy(f"{I}/{f}", f"{W}/scripts/{f}")
os.symlink(f"{I}/details_all.jsonl", f"{W}/data/eval/details_all.jsonl")
for f in ("e5_queries.npy", "e5_qids.json"): os.symlink(f"{I}/{f}", f"{W}/e5/{f}")
subprocess.run("pip install -q pandas==3.0.6 numpy==2.5.3 lightgbm==4.7.0 2>&1 | tail -2", shell=True)
subprocess.run("pip list 2>/dev/null | grep -iE '^(pandas|lightgbm|numpy) '", shell=True)
env = dict(os.environ, R2AI_LTR_PARAMS='{"learning_rate":0.03,"lambda_l2":10,"min_data_in_leaf":50}')
r = subprocess.run([sys.executable, "scripts/ltr.py", "--preds", "x", "--scores", "x", "--dense", "x", "--xq", "e5", "--xcache", f"{I}/x.pkl",
                    "--rounds", "700", "--stages", "1", "--seeds", "5", "--out", "ltr_order_oof28.json"], cwd=W, env=env)
shutil.rmtree(f"{W}/scripts"); shutil.rmtree(f"{W}/data"); shutil.rmtree(f"{W}/e5")
print("EXIT", r.returncode)
