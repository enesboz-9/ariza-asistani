"""Terminalde arıza tahmini sohbeti.   python chat.py

Karar küçük ML modelinden gelir; cevap şablondur (uydurma riski yok).
Model emin değilse, en olası arızaları ayıran bir soru sorar.
"""
import json
import sys
from datetime import datetime

import joblib

from common import DATA, MODELS, load_kb

CONF = 0.45       # bu olasılığın altında soru sor
MARGIN = 0.15     # 1. ve 2. arıza arası fark bunun altındaysa soru sor
MAX_QUESTIONS = 2
DISCLAIMER = "Bu bir ön tahmindir; kesin teşhis için yetkili servise danışın."


def predict(model, text):
    probs = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])
    return ranked


def ask_followup(kb, ranked, asked):
    """En olası iki arızanın henüz sorulmamış bir belirtisini sor."""
    for label, _ in ranked[:2]:
        for symptom in kb[label]["belirtiler"]:
            if symptom not in asked:
                return symptom
    return None


def show(kb, ranked):
    print("\nOlası arızalar:")
    for label, p in ranked[:3]:
        print(f"  %{p * 100:4.0f}  {kb[label]['ad']}")
    top = kb[ranked[0][0]]
    print(f"\nÖneri: {top['oneri']}")
    if top["oncelik"] == "yüksek":
        print("DİKKAT: Bu arıza güvenlik riski taşıyabilir. Mümkünse aracı kullanmadan servise götürün.")
    print(DISCLAIMER)


def save_feedback(text, ranked):
    """Doğrulanmamış geri bildirim; sen kontrol edip feedback_verified.jsonl'e taşırsın."""
    answer = input("Tahmin doğru muydu? (e/h, geç için Enter): ").strip().lower()
    if answer not in ("e", "h"):
        return
    row = {"text": text, "predicted": ranked[0][0], "correct": answer == "e",
           "time": datetime.now().isoformat(timespec="seconds")}
    if answer == "h":
        row["label"] = input("Gerçek arıza kodu (ör. fren, aku, sogutma; bilmiyorsan boş bırak): ").strip()
    with open(DATA / "feedback_pending.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    path = MODELS / "ariza_model.joblib"
    if not path.exists():
        sys.exit("Model yok. Önce: python run_pipeline.py")
    model, kb = joblib.load(path), load_kb()
    print("Araç arıza asistanı. Aracınızdaki belirtiyi anlatın (çıkmak için 'q').\n")
    while True:
        text = input("Sen: ").strip()
        if text.lower() in ("q", "çık", "cik", "exit"):
            break
        if not text:
            continue
        asked, questions = set(), 0
        ranked = predict(model, text)
        while (ranked[0][1] < CONF or ranked[0][1] - ranked[1][1] < MARGIN) and questions < MAX_QUESTIONS:
            symptom = ask_followup(kb, ranked, asked)
            if symptom is None:
                break
            asked.add(symptom)
            questions += 1
            reply = input(f"Bot: Şunu da yaşıyor musunuz: \"{symptom}\"? (e/h): ").strip().lower()
            if reply.startswith("e"):
                text += ". " + symptom
                ranked = predict(model, text)
        show(kb, ranked)
        save_feedback(text, ranked)
        print()


if __name__ == "__main__":
    main()
