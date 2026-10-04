"""Küçük modeli eğitir; yalnızca eskisinden kötü değilse kaydeder.

Eğitim verisi: data/synthetic.jsonl + data/feedback_verified.jsonl
(kullanıcı geri bildirimleri, senin onayladığın satırlar).
Ölçüm: data/test_real.jsonl (elle yazılmış, eğitime hiç girmez).
"""
import hashlib
import json
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from common import DATA, MODELS, read_jsonl, strip_diacritics, tr_lower

MODEL_PATH = MODELS / "ariza_model.joblib"
META_PATH = MODELS / "meta.json"


def build():
    return make_pipeline(
        TfidfVectorizer(preprocessor=tr_lower, analyzer="char_wb",
                        ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                        max_features=30000),
        LogisticRegression(C=8, max_iter=2000),
    )


def main() -> int:
    sources = {
        "sentetik": read_jsonl(DATA / "synthetic.jsonl"),
        "dış veri (CSV)": read_jsonl(DATA / "external.jsonl"),
        "onaylı geri bildirim": read_jsonl(DATA / "feedback_verified.jsonl"),
        "öz-oyun (LLM sohbetleri)": read_jsonl(DATA / "selfplay.jsonl"),
    }
    train = [r for rows in sources.values() for r in rows]
    for name, rows in sources.items():
        print(f"  {name}: {len(rows)} cümle")
    test = read_jsonl(DATA / "test_real.jsonl")
    if not train:
        print("Eğitim verisi yok. Önce generate_data.py çalıştır.")
        return 1

    model = build()
    model.fit([r["text"] for r in train], [r["label"] for r in train])
    pred = model.predict([r["text"] for r in test])
    y = [r["label"] for r in test]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"Eğitim: {len(train)} cümle | Test: {len(test)} cümle")
    print(f"Doğruluk: {acc:.3f} | Makro F1: {f1:.3f}")
    ascii_pred = model.predict([strip_diacritics(r["text"]) for r in test])
    print(f"Türkçe karaktersiz yazımda doğruluk: {accuracy_score(y, ascii_pred):.3f}")

    # Test seti değiştiyse eski skor karşılaştırılamaz: kapı yalnızca aynı test setinde çalışır.
    test_id = hashlib.sha1(
        "\n".join(sorted(r["text"] + r["label"] for r in test)).encode()
    ).hexdigest()[:12]
    old = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
    if old and old.get("test_id") != test_id:
        print("Test seti değişmiş; eski skorla karşılaştırma yapılmadı.")
        old = None
    if old and f1 < old["macro_f1"]:
        print(f"Yeni model eskisinden kötü ({f1:.3f} < {old['macro_f1']:.3f}). Eski model korundu.")
        return 0

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps(
        {"accuracy": acc, "macro_f1": f1, "train_size": len(train), "test_id": test_id}, indent=2))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Model kaydedildi: {MODEL_PATH.name} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
