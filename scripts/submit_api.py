"""Nộp bài lên leaderboard Codabench bằng API (không cần trình duyệt) + chờ chấm + tải scoring_result.

    python scripts/submit_api.py submissions/v25.zip [--phase 44] [--wait]
Tài khoản đọc từ .r2ai_cred (R2AI_USER / R2AI_PASS). Kết quả chi tiết lưu data/eval/details_<id>_<tên>.jsonl.
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile

import requests

BASE = "https://leaderboard.aiguru.com.vn"


def session():
    cred = dict(l.strip().split("=", 1) for l in open(".r2ai_cred") if "=" in l)
    s = requests.Session()
    r = s.get(f"{BASE}/accounts/login/")
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    s.post(f"{BASE}/accounts/login/", data={"csrfmiddlewaretoken": tok, "username": cred["R2AI_USER"],
                                           "password": cred["R2AI_PASS"]}, headers={"Referer": f"{BASE}/accounts/login/"})
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": f"{BASE}/competitions/15/"})
    return s


def submit(s, path, phase):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    for attempt in range(6):   # PUT lớn đôi khi bị SSLEOF -> xin URL mới và thử lại
        nm = name if attempt == 0 else name.replace(".zip", f"_r{int(time.time())}.zip")   # tên trùng -> bị từ chối
        d = s.post(f"{BASE}/api/datasets/", json={"type": "submission", "request_sassy_file_name": nm, "name": nm,
                                                 "description": "", "is_public": False, "file_size": size}).json()
        if "key" not in d:
            print("dataset create rejected:", str(d)[:300], flush=True)
            continue
        key, url = d["key"], d["sassy_url"]
        try:
            with open(path, "rb") as f:
                r = requests.put(url, data=f, headers={"Content-Type": "application/zip", "Content-Length": str(size)},
                                 timeout=3600)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"upload attempt {attempt + 1} failed: {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        sys.exit("upload failed")
    s.put(f"{BASE}/api/datasets/completed/{key}/").raise_for_status()
    comp = s.get(f"{BASE}/api/competitions/15/").json()
    tasks = [t["id"] if isinstance(t, dict) else t for p in comp["phases"] if p["id"] == phase for t in p.get("tasks", [])]
    r = s.post(f"{BASE}/api/submissions/", json={"data": key, "phase": phase, "tasks": tasks, "fact_sheet_answers": None})
    r.raise_for_status()
    return r.json()["id"]


def wait(s, sid, name):
    while True:
        j = s.get(f"{BASE}/api/submissions/?phase__competition=15").json()
        st = next(x for x in (j.get("results") or j) if x["id"] == sid)["status"]
        if st in ("Finished", "Failed", "Cancelled"):
            break
        time.sleep(20)
    if st != "Finished":
        print("status", st); return
    d = s.get(f"{BASE}/api/submissions/{sid}/get_details/").json()
    z = zipfile.ZipFile(io.BytesIO(requests.get(d["scoring_result"], timeout=300).content))
    print(z.read("scores.txt").decode())
    os.makedirs("data/eval", exist_ok=True)
    open(f"data/eval/details_{sid}_{name}.jsonl", "wb").write(z.read("details.jsonl"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip"); ap.add_argument("--phase", type=int, default=44); ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    s = session()
    sid = submit(s, a.zip, a.phase)
    print("submitted id", sid, flush=True)
    if a.wait:
        wait(s, sid, os.path.basename(a.zip).replace(".zip", ""))
