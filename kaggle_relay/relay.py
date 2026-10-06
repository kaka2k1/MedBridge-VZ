"""Kaggle CPU: PUT file zip bài nộp lên presigned URL của Codabench (mạng Kaggle→BTC nhanh hơn mạng local)."""
import glob, subprocess
z = glob.glob("/kaggle/input/**/sub.bin", recursive=True)[0]
url = open(glob.glob("/kaggle/input/**/url.txt", recursive=True)[0]).read().strip()
r = subprocess.run(["curl", "-sS", "-T", z, "-H", "Content-Type: application/zip", "-o", "/dev/null", "-w", "%{http_code} %{speed_upload}", url],
                   capture_output=True, text=True)
open("/kaggle/working/put_result.txt", "w").write(r.stdout + " " + r.stderr[:200])
print(r.stdout, r.stderr[:200])
