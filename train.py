"""Küçük modeli eğitir; yalnızca eskisinden kötü değilse kaydeder.

Eğitim verisi: data/synthetic.jsonl + data/feedback_verified.jsonl
(kullanıcı geri bildirimleri, senin onayladığın satırlar).
Ölçüm: data/test_real.jsonl (elle yazılmış, eğitime hiç girmez).
"""
import collections
import hashlib
import json
import random
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline, make_union

from common import DATA, MODELS, read_jsonl, strip_diacritics, tr_lower

# Windows konsol/yönlendirme kodlaması ne olursa olsun Türkçe ve simgeler çökmesin
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_PATH = MODELS / "ariza_model.joblib"
META_PATH = MODELS / "meta.json"


# Şablon verisi aynı ~150 ifadenin kombinasyonlarıdır; LLM/öz-oyun/geri bildirim cümleleri ise
# gerçek dile yakın ve az sayıdadır. Ağırlık vermezsek 8000 şablon cümle 5-10 yeni cümleyi bastırır.
SOURCE_WEIGHT = {"sentetik": 1.0, "dış veri (CSV)": 3.0,
                 "onaylı geri bildirim": 5.0, "öz-oyun (LLM sohbetleri)": 5.0}

# Yeni model, şimdiye kadarki EN İYİ skordan bu kadar kötüyse de kabul edilir. 84 cümlelik testte
# tek cümle ~%1.2 eder; 0.01'den küçük farklar gürültüdür. En iyi skorla kıyaslandığı için
# art arda küçük kayıplarla sessizce kötüleşme (drift) olmaz.
TOLERANCE = 0.01


def build():
    return make_pipeline(
        make_union(
            TfidfVectorizer(preprocessor=tr_lower, analyzer="char_wb",
                            ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=30000),
            TfidfVectorizer(preprocessor=tr_lower, analyzer="word",
                            ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        ),
        LogisticRegression(C=8, max_iter=2000),
    )


def bootstrap_ci(y, pred, n=500, seed=0):
    """Test seti küçük olduğu için doğruluğun %95 güven aralığı (kaba)."""
    rng = random.Random(seed)
    hits = [int(a == b) for a, b in zip(y, pred)]
    accs = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(n))
    return accs[int(n * 0.025)], accs[int(n * 0.975)]


def main() -> int:
    sources = {
        "sentetik": read_jsonl(DATA / "synthetic.jsonl"),
        "dış veri (CSV)": read_jsonl(DATA / "external.jsonl"),
        "onaylı geri bildirim": read_jsonl(DATA / "feedback_verified.jsonl"),
        "öz-oyun (LLM sohbetleri)": read_jsonl(DATA / "selfplay.jsonl"),
    }
    train, weights = [], []
    for name, rows in sources.items():
        train += rows
        weights += [SOURCE_WEIGHT[name]] * len(rows)
    for name, rows in sources.items():
        print(f"  {name}: {len(rows)} cümle")
    test = read_jsonl(DATA / "test_real.jsonl")
    if not train:
        print("Eğitim verisi yok. Önce generate_data.py çalıştır.")
        return 1

    model = build()
    model.fit([r["text"] for r in train], [r["label"] for r in train],
              logisticregression__sample_weight=weights)
    pred = model.predict([r["text"] for r in test])
    y = [r["label"] for r in test]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    print(f"Eğitim: {len(train)} cümle | Test: {len(test)} cümle")
    print(f"Doğruluk: {acc:.3f} | Makro F1: {f1:.3f}")
    ascii_pred = model.predict([strip_diacritics(r["text"]) for r in test])
    print(f"Türkçe karaktersiz yazımda doğruluk: {accuracy_score(y, ascii_pred):.3f}")
    lo, hi = bootstrap_ci(y, pred)
    print(f"Doğruluk %95 aralığı (bootstrap): {lo:.2f} - {hi:.2f}  <- test küçük, küçük farklar anlamsız")
    pairs = collections.Counter((a, b) for a, b in zip(y, pred) if a != b)
    if pairs:
        print("En sık karışanlar (gerçek -> tahmin): " +
              ", ".join(f"{a}->{b} x{n}" for (a, b), n in pairs.most_common(5)))

    # Test seti değiştiyse eski skor karşılaştırılamaz: kapı yalnızca aynı test setinde çalışır.
    test_id = hashlib.sha1(
        "\n".join(sorted(r["text"] + r["label"] for r in test)).encode()
    ).hexdigest()[:12]
    old = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
    if old and old.get("test_id") != test_id:
        print("Test seti değişmiş; eski skorla karşılaştırma yapılmadı.")
        old = None
    best = max(old.get("best_f1", 0), old["macro_f1"]) if old else 0.0
    if old and f1 < best - TOLERANCE:
        print(f"Yeni model belirgin şekilde kötü ({f1:.3f} < en iyi {best:.3f} - {TOLERANCE}). Eski model korundu.")
        return 0
    if old and f1 < old["macro_f1"]:
        print(f"Küçük düşüş ({f1:.3f} < {old['macro_f1']:.3f}) tolerans içinde, yeni model kabul edildi.")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps(
        {"accuracy": acc, "macro_f1": f1, "best_f1": max(best, f1),
         "train_size": len(train), "test_id": test_id}, indent=2))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Model kaydedildi: {MODEL_PATH.name} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
