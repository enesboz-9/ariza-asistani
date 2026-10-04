"""Tek komutla: veri üret -> eğit -> (iyiyse) kaydet.

  python run_pipeline.py              # çevrimdışı şablon verisiyle
  python run_pipeline.py --llm        # Claude API ile üretilmiş veriyle
  python run_pipeline.py --llm --per-class 200

Zamanlayıcıyla (Windows Görev Zamanlayıcı / cron) periyodik çalıştırılabilir.
"""
import subprocess
import sys

args = sys.argv[1:]
mode = "llm" if "--llm" in args else "template"
extra = [a for a in args if a != "--llm"]
py = sys.executable
subprocess.run([py, "generate_data.py", "--mode", mode, *extra], check=True)
sys.exit(subprocess.run([py, "train.py"]).returncode)
