#!/bin/bash
# Vòng private 04/11: nộp 3 bài thêm (P8, P10 từ Kaggle; P9 local) NGAY khi nén xong, KHÔNG chờ điểm.
# An toàn: không nộp lại file đã có submitted id; dừng nếu đã dùng >= 4/5 lượt (giữ 1 lượt dự phòng);
# 22:45 nếu chưa đủ 3 bài thì nộp P6_v74.zip (có sẵn) bù chỗ thiếu.
cd /home/phong/Desktop/hsk/test2
L=data/cache/priv2/chain.log; DONE=data/cache/priv2/submitted.txt; touch $DONE
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
used() { timeout 120 .venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from submit_api import session, BASE
c = session().get(f"{BASE}/api/competitions/15/", timeout=60).json()
print([p for p in c["phases"] if p["id"] == 45][0]["used_submissions_per_person"])
PY
}
check() {  # $1 = zip: đúng 1 tệp submission.json, 1200 câu, < 100 MB
  timeout 600 .venv/bin/python -I - "$1" <<'PY'
import json, os, sys, zipfile
f = sys.argv[1]; n = os.path.getsize(f); z = zipfile.ZipFile(f)
assert z.namelist() == ["submission.json"], z.namelist()
d = json.loads(z.read("submission.json")); assert len(d) == 1200 and len({r["id"] for r in d}) == 1200
assert n < 100_000_000, n
print("OK", n, "docs/q", round(sum(len(r["relevant_docs"]) for r in d) / 1200, 1), "chunks/q", round(sum(len(r["relevant_chunks"]) for r in d) / 1200, 1))
PY
}
submit() {  # $1 = zip
  local Z=$1 B=$(basename $1 .zip)
  grep -q "^$B " $DONE && { log "$B đã nộp – bỏ qua"; return 0; }
  local u=$(used); log "đã dùng $u/5 lượt"
  [ -n "$u" ] && [ "$u" -ge 4 ] && { log "ĐÃ DÙNG >= 4 lượt – dừng, không nộp $B"; return 1; }
  check "$Z" >> $L 2>&1 || { log "$B KIỂM TRA LỖI – không nộp"; return 1; }
  cp -n "$Z" submissions/private/ && (sha256sum submissions/private/$B.zip >> submissions/private/SHA256SUMS)
  log "nộp $B (relay, phase 45)"
  .venv/bin/python scripts/submit_api.py "$Z" --phase 45 --relay > data/cache/priv2/sub_$B.log 2>&1
  local id=$(grep -oE "submitted id [0-9]+" data/cache/priv2/sub_$B.log | grep -oE "[0-9]+")
  if [ -n "$id" ]; then echo "$B $id" >> $DONE; log "$B → submitted id $id"; else log "$B relay lỗi: $(tail -3 data/cache/priv2/sub_$B.log | tr '\n' ' ')"; fi
}
Q="data/cache/priv2/kg/P8_v80a_s10.zip data/cache/priv2/local/P8_v80a_s10.zip data/cache/priv2/kg/P10_ef15.zip data/cache/priv2/local/P10_ef15.zip data/cache/priv2/kg9/P9_v75b.zip"
fetch() {  # $1 kernel, $2 thư mục: tải output khi kernel xong (1 lần)
  [ -e $2/.fetched ] && return
  st=$(timeout 120 .venv/bin/kaggle kernels status $1 2>/dev/null | grep -oiE "complete|error|cancel")
  [ -z "$st" ] && return
  log "kernel $1: $st"; mkdir -p $2.tmp
  timeout 1800 .venv/bin/kaggle kernels output $1 -p $2.tmp >> $L 2>&1 && rm -rf $2 && mv $2.tmp $2 && touch $2/.fetched && log "tải xong $1: $(ls $2 | tr '\n' ' ')"
}
deadline=$(date -d "today 22:45" +%s)
while :; do
  n=$(wc -l < $DONE); [ "$n" -ge 3 ] && { log "đủ 3 bài: $(tr '\n' ';' < $DONE)"; break; }
  fetch kakaoffical/r2ai-buildzip data/cache/priv2/kg; fetch kakaoffical/r2ai-buildzip-p9 data/cache/priv2/kg9
  for z in $Q; do
    b=$(basename $z .zip)
    grep -q "^$b " $DONE && continue
    [ -s $z ] || continue
    submit $z
  done
  if [ $(date +%s) -ge $deadline ] && [ $(wc -l < $DONE) -lt 3 ]; then
    log "22:45 – chưa đủ 3 bài, nộp P6_v74 bù"; submit submissions/private/P6_v74.zip; break
  fi
  sleep 60
done
log "kết thúc"
