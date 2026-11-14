#!/bin/bash
# Chạy ViBioMIR Live: bash app_live/run.sh  → tự mở http://localhost:8765 khi sẵn sàng
#   PORT=8800 bash app_live/run.sh      đổi cổng
#   NO_BROWSER=1 bash app_live/run.sh   không tự mở trình duyệt
cd "$(dirname "$0")/.." || exit 1
PORT="${PORT:-8765}"; URL="http://localhost:$PORT"
G=$'\e[32m'; RED=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'
say() { echo "${B}[ViBioMIR]${N} $*"; }
open_browser() { [ -z "$NO_BROWSER" ] && command -v xdg-open >/dev/null && (xdg-open "$URL" >/dev/null 2>&1 &); }

[ -x .venv/bin/python ] || { say "${RED}Không thấy .venv/bin/python – hãy tạo môi trường ảo và cài requirements.txt${N}"; exit 1; }
for f in app_live/docs.sqlite data/cache/dense_gn data/cache/zhfeat3_out/zh_emb.npy data/models/bge-m3-st; do
  [ -e "$f" ] || { say "${RED}Thiếu dữ liệu: $f (xem app_live/README.md)${N}"; exit 1; }
done
if curl -s -m 2 "$URL/api/health" >/dev/null 2>&1; then
  say "${G}Máy chủ đã chạy sẵn tại $URL${N}"; open_browser; exit 0
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  say "${RED}Cổng $PORT đang bị chương trình khác dùng. Thử: PORT=8800 bash app_live/run.sh${N}"; exit 1
fi
command -v nvidia-smi >/dev/null && say "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)"
say "Đang nạp chỉ mục và 4 mô hình (~1 phút). Trình duyệt sẽ tự mở khi sẵn sàng…"

export PORT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
.venv/bin/python -u app_live/server.py > >(tee app_live/server.log) 2>&1 &
SRV=$!
trap 'echo; say "Đang tắt máy chủ…"; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 0' INT TERM

T0=$SECONDS
until curl -s -m 2 "$URL/api/health" >/dev/null 2>&1; do
  kill -0 $SRV 2>/dev/null || { say "${RED}Máy chủ dừng khi khởi động – xem app_live/server.log${N}"; exit 1; }
  sleep 2
done
say "${G}${B}Sẵn sàng sau $((SECONDS-T0)) giây → $URL${N}   (Ctrl+C để tắt)"
open_browser
wait $SRV
