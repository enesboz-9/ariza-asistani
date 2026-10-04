"""Arıza botu için tek dosyalık araç kutusu.  ariza_bot klasöründen çalıştır:

  python araclar.py test                 # otomatik testler (veri, sızıntı, model, soru seçimi)
  python araclar.py degerlendir [--chat] [--file x.jsonl]   # sınıf raporu, karışanlar, yanlışlar
  python araclar.py ekle                 # elle test cümlesi ekle (etkileşimli)
  python araclar.py ekle --file yeni.txt # toplu: satır = arıza_kodu<TAB>cümle  (ya da  kod | cümle)
  python araclar.py durum                # arıza başına test cümlesi sayısı
  python araclar.py agirlik              # öz-oyun ağırlığını 5 farklı veri çekilişiyle dene (~1-2 dk)
  python araclar.py denetle              # öz-oyun cümlelerinde şüpheli etiketleri listele (silmez)
  python araclar.py yama                 # train.py ağırlığı + generate_data.py sıralama hatası düzeltmesi (yedek alır)

Gerekenler: ariza_bot klasöründeki chat.py, common.py, train.py (zaten var).
"""
import argparse
import collections
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from common import DATA, MODELS, load_kb, read_jsonl, strip_diacritics, tr_lower, write_jsonl  # noqa: E402

TEST_PATH = DATA / "test_real.jsonl"
SELF_KEY = "öz-oyun (LLM sohbetleri)"
TRAIN_FILES = ("synthetic", "external", "feedback_verified", "selfplay")


# ====================================================================== durum / ekle

def counts(rows, kb):
    c = {k: 0 for k in kb}
    for r in rows:
        c[r["label"]] = c.get(r["label"], 0) + 1
    return c


def cmd_durum(_args=None):
    kb, rows = load_kb(), read_jsonl(TEST_PATH)
    print(f"Toplam test cümlesi: {len(rows)}")
    for label, n in sorted(counts(rows, kb).items(), key=lambda x: x[1]):
        print(f"  {label:12s} {'#' * n} {n}")


def remove_from_training(text: str) -> int:
    """Teste eklenen cümleyi eğitim dosyalarından siler (sızıntı olmasın)."""
    key, removed = tr_lower(text), 0
    for name in TRAIN_FILES:
        path = DATA / f"{name}.jsonl"
        rows = read_jsonl(path)
        kept = [r for r in rows if tr_lower(r["text"]) != key]
        if len(kept) != len(rows):
            write_jsonl(path, kept)
            removed += len(rows) - len(kept)
    return removed


def try_add(rows, text, label, kb):
    """Eklenebiliyorsa None, eklenemiyorsa nedenini döndürür."""
    text = " ".join(text.split())
    if label not in kb:
        return f"bilinmeyen arıza kodu '{label}'"
    if len(text.split()) < 3:
        return "cümle çok kısa"
    if tr_lower(text) in {tr_lower(r["text"]) for r in rows}:
        return "bu cümle testte zaten var"
    if tr_lower(text) in {tr_lower(b) for info in kb.values() for b in info["belirtiler"]}:
        return "bilgi tabanındaki belirti cümlesinin aynısı; kendi sözlerinle yaz"
    rows.append({"text": text, "label": label})
    remove_from_training(text)
    return None


def cmd_ekle(args):
    kb, rows = load_kb(), read_jsonl(TEST_PATH)
    labels = list(kb)
    added = 0
    if args.file:
        for n, line in enumerate(open(args.file, encoding="utf-8"), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "\t" if "\t" in line else "|"
            if sep not in line:
                print(f"  satır {n}: ayraç yok, atlandı")
                continue
            label, text = (x.strip() for x in line.split(sep, 1))
            err = try_add(rows, text, label.lower(), kb)
            print(f"  satır {n}: eklenmedi ({err})") if err else None
            added += err is None
        write_jsonl(TEST_PATH, rows)
        print(f"{added} cümle eklendi. Toplam: {len(rows)}")
        return
    cmd_durum()
    print("\nArıza kodları:\n  " + ", ".join(labels))
    print("Boş cümle = çık. Kodu yarım yazman yeter (ör. 'sog' -> sogutma).\n")
    while True:
        text = input("Cümle: ").strip()
        if not text:
            break
        code = input("Arıza kodu: ").strip().lower()
        matches = ([l for l in labels if l == code] or [l for l in labels if l.startswith(code)]) if code else []
        if len(matches) != 1:
            print(f"  anlaşılmadı ({', '.join(matches) or 'eşleşme yok'}); atlandı.\n")
            continue
        err = try_add(rows, text, matches[0], kb)
        if err:
            print(f"  eklenmedi: {err}\n")
            continue
        added += 1
        write_jsonl(TEST_PATH, rows)               # her eklemede kaydet
        c = counts(rows, kb)
        print(f"  eklendi ({matches[0]}: {c[matches[0]]}). En az cümlesi olanlar: "
              f"{', '.join(sorted(c, key=c.get)[:3])}\n")
    print(f"{added} cümle eklendi. Toplam: {len(rows)}")


# ====================================================================== değerlendir

def cmd_degerlendir(args):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from chat import CONF, MARGIN, MAX_QUESTIONS, ask_followup, load_model, predict

    model, kb = load_model(), load_kb()
    rows = read_jsonl(Path(args.file) if args.file else TEST_PATH)
    texts, y = [r["text"] for r in rows], [r["label"] for r in rows]
    pred = list(model.predict(texts))

    print(classification_report(y, pred, zero_division=0))
    top3 = sum(g in [l for l, _ in predict(model, t)[:3]] for t, g in zip(texts, y)) / len(y)
    print(f"Doğruluk: {accuracy_score(y, pred):.3f} | Makro F1: {f1_score(y, pred, average='macro'):.3f} "
          f"| İlk-3 doğruluk: {top3:.3f}")
    print(f"Türkçe karaktersiz: {accuracy_score(y, model.predict([strip_diacritics(t) for t in texts])):.3f}")
    low = model.predict([t.lower().replace('.', '').replace(',', '') for t in texts])
    print(f"Küçük harf + noktalamasız: {accuracy_score(y, low):.3f}")

    conf = [predict(model, t)[0][1] for t in texts]
    right = [c for c, a, b in zip(conf, y, pred) if a == b]
    wrong = [c for c, a, b in zip(conf, y, pred) if a != b]
    if right and wrong:
        print(f"Ortalama güven: doğrularda {sum(right) / len(right):.2f}, yanlışlarda "
              f"{sum(wrong) / len(wrong):.2f} (fark küçükse CONF/MARGIN eşikleri ayarlanmalı)")
    pairs = collections.Counter((a, b) for a, b in zip(y, pred) if a != b)
    if pairs:
        print("\nEn sık karışanlar:")
        for (a, b), n in pairs.most_common(8):
            print(f"  {a} -> {b}  x{n}")
        print("\nYanlış tahminler:")
        for t, a, b, c in zip(texts, y, pred, conf):
            if a != b:
                print(f"  [{a} -> {b}, güven {c:.2f}] {t}")

    if args.chat:
        ok = asked_total = 0
        for r in rows:
            text, asked, q = r["text"], set(), 0
            ranked = predict(model, text)
            while (ranked[0][1] < CONF or ranked[0][1] - ranked[1][1] < MARGIN) and q < MAX_QUESTIONS:
                try:
                    symptom = ask_followup(kb, ranked, asked, model)
                except TypeError:                      # eski chat.py (model parametresi yok)
                    symptom = ask_followup(kb, ranked, asked)
                if symptom is None:
                    break
                asked.add(symptom)
                q += 1
                if symptom in kb[r["label"]]["belirtiler"]:
                    text += ". " + symptom
                    ranked = predict(model, text)
            ok += ranked[0][0] == r["label"]
            asked_total += q
        print(f"\nSohbet akışıyla doğruluk: {ok / len(rows):.3f} (ortalama {asked_total / len(rows):.2f} soru)")


# ====================================================================== ağırlık / denetle / yama

def synth(kb, seed, banned):
    """Bellekte, deterministik sentetik veri üretir (generate_data ile aynı yöntem)."""
    import random
    from common import augment
    from generate_data import template_sentences
    rng = random.Random(seed)
    rows = template_sentences(kb, 300, rng)
    rows.sort(key=lambda r: (r["label"], r["text"]))          # set sırasından bağımsız
    rows += [{"text": augment(r["text"], rng), "label": r["label"]} for r in rows if rng.random() < 0.5]
    out, seen = [], set()
    for r in rows:
        k = tr_lower(r["text"])
        if k not in banned and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def cmd_agirlik(args=None):
    """Öz-oyun ağırlığını, birkaç farklı sentetik veri çekilişi üzerinden ortalayarak dener."""
    from sklearn.metrics import accuracy_score, f1_score
    from train import SOURCE_WEIGHT, build

    kb, test = load_kb(), read_jsonl(TEST_PATH)
    banned = {tr_lower(r["text"]) for r in test}
    others = {n: [r for r in read_jsonl(DATA / f"{f}.jsonl") if tr_lower(r["text"]) not in banned]
              for n, f in (("dış veri (CSV)", "external"), ("onaylı geri bildirim", "feedback_verified"))}
    selfplay = [r for r in read_jsonl(DATA / "selfplay.jsonl") if tr_lower(r["text"]) not in banned]
    if not selfplay:
        sys.exit("data/selfplay.jsonl boş ya da yok; önce python self_play.py çalıştır.")
    seeds, weights_try = (1, 2, 3, 4, 5), (0, 1, 2, 5)
    print(f"öz-oyun cümlesi: {len(selfplay)} | test: {len(test)} cümle | {len(seeds)} farklı sentetik veri çekilişi")
    print("(her ağırlık için 5 çekilişin ortalaması; ~1-2 dk sürer)\n")
    y = [r["label"] for r in test]
    results = {w: [] for w in weights_try}
    for seed in seeds:
        syn = synth(kb, seed, banned)
        for w in weights_try:
            rows = syn + others["dış veri (CSV)"] + others["onaylı geri bildirim"] + selfplay
            wts = ([SOURCE_WEIGHT["sentetik"]] * len(syn)
                   + [SOURCE_WEIGHT["dış veri (CSV)"]] * len(others["dış veri (CSV)"])
                   + [SOURCE_WEIGHT["onaylı geri bildirim"]] * len(others["onaylı geri bildirim"])
                   + [w] * len(selfplay))
            m = build().fit([r["text"] for r in rows], [r["label"] for r in rows],
                            logisticregression__sample_weight=wts)
            p = m.predict([r["text"] for r in test])
            results[w].append((accuracy_score(y, p), f1_score(y, p, average="macro")))
        print(f"  çekiliş {seed}/{len(seeds)} bitti", flush=True)
    print("\nağırlık  ort.doğruluk  en düşük-en yüksek   ort.makro-F1")
    best = None
    for w, rs in results.items():
        accs, f1s = [a for a, _ in rs], [f for _, f in rs]
        mean_f1 = sum(f1s) / len(f1s)
        print(f"{w:6}   {sum(accs) / len(accs):.3f}         {min(accs):.3f}-{max(accs):.3f}         {mean_f1:.3f}")
        if best is None or mean_f1 > best[1] + 1e-9:
            best = (w, mean_f1)
    base = sum(f for _, f in results[0]) / len(seeds)
    gain = best[1] - base
    print(f"\nEn yüksek ortalama F1: ağırlık {best[0]} (ağırlık 0'a göre {gain:+.3f}).")
    if best[0] == 0 or gain < 0.02:
        print("Fark küçük: öz-oyun verisi bu testte ölçülebilir fayda sağlamıyor. Ağırlığı 0-1 tut,\n"
              "öz-oyun yerine test setini büyüt (python araclar.py ekle).")
    else:
        print(f"Öz-oyun yardım ediyor. train.py'de öz-oyun ağırlığını {best[0]} yap.")


def cmd_denetle(_args=None):
    from train import build

    kb, rows = load_kb(), read_jsonl(DATA / "selfplay.jsonl")
    if not rows:
        sys.exit("data/selfplay.jsonl boş ya da yok.")
    base = [r for f in ("synthetic", "external", "feedback_verified") for r in read_jsonl(DATA / f"{f}.jsonl")]
    model = build().fit([r["text"] for r in base], [r["label"] for r in base])
    classes = list(model.classes_)
    probs = model.predict_proba([r["text"] for r in rows])
    suspects = []
    for i, (r, p) in enumerate(zip(rows, probs), 1):
        text, why = tr_lower(r["text"]), []
        own = any(tr_lower(k) in text for k in kb[r["label"]].get("anahtar_kelimeler", []))
        others = sorted({lab for lab, info in kb.items() if lab != r["label"]
                         for k in info.get("anahtar_kelimeler", []) if tr_lower(k) in text})
        if others and not own:
            why.append("başka arızanın parça adı geçiyor: " + ", ".join(others))
        top = max(range(len(p)), key=p.__getitem__)
        if classes[top] != r["label"] and p[top] >= 0.8:
            why.append(f"model {classes[top]} diyor (güven {p[top]:.2f})")
        if why:
            suspects.append({"satir": i, "label": r["label"], "text": r["text"], "neden": "; ".join(why)})
    print(f"{len(rows)} öz-oyun cümlesinden {len(suspects)} şüpheli (%{100 * len(suspects) / len(rows):.0f})\n")
    for s in suspects[:40]:
        print(f"satır {s['satir']:4d} [{s['label']}] {s['text']}\n          -> {s['neden']}")
    if len(suspects) > 40:
        print(f"... ve {len(suspects) - 40} tane daha (hepsi data/selfplay_suspect.jsonl içinde)")
    write_jsonl(DATA / "selfplay_suspect.jsonl", suspects)
    print("\nBunlar sezgisel işaretlerdir, yanlış alarm olabilir. Gerçekten yanlışsa o satırı "
          "data/selfplay.jsonl içinden elle sil. Zor ama doğru cümleleri silme.")


def cmd_yama(_args=None):
    """İki düzeltme: (1) öz-oyun ağırlığı 5 -> 2, (2) generate_data.py'de rastgele set sırası hatası.
    (2) olmadan her çalıştırma farklı eğitim verisi üretir ve doğruluk 0.87-0.88 arasında oynar."""
    done = []
    tp = ROOT / "train.py"
    src = tp.read_text(encoding="utf-8")
    new, n = re.subn(r'("öz-oyun \(LLM sohbetleri\)"\s*:\s*)5\.0', r"\g<1>2.0", src)
    if n:
        (ROOT / "train.py.yedek").write_text(src, encoding="utf-8")
        tp.write_text(new, encoding="utf-8")
        done.append("train.py: öz-oyun ağırlığı 5.0 -> 2.0 (yedek: train.py.yedek)")

    gp = ROOT / "generate_data.py"
    src = gp.read_text(encoding="utf-8")
    old = 'rows += [{"text": t, "label": label} for t in seen]'
    if old in src:
        (ROOT / "generate_data.py.yedek").write_text(src, encoding="utf-8")
        gp.write_text(src.replace(old, 'rows += [{"text": t, "label": label} for t in sorted(seen)]'),
                      encoding="utf-8")
        done.append("generate_data.py: cümleler sıralı yazılıyor (artık aynı --seed aynı veriyi üretir)")

    meta = MODELS / "meta.json"
    if meta.exists() and done:
        meta.rename(MODELS / "meta.json.yedek")
        done.append("models/meta.json yedeklendi: eski 'en iyi skor' şanslı bir veri çekilişinden "
                    "geliyordu, bir sonraki eğitim yeni taban çizgisini kurar")
    print("\n".join("- " + d for d in done) if done else "Yapılacak yama kalmadı (zaten uygulanmış olabilir).")
    if done:
        print("\nSıradaki adım: python run_pipeline.py")


# ====================================================================== testler

class TestCommon(unittest.TestCase):
    def test_tr_lower(self):
        self.assertEqual(tr_lower("ISINMA İZİ"), "ısınma izi")

    def test_strip_diacritics(self):
        self.assertEqual(strip_diacritics("öğrenci şoförü çağırdı"), "ogrenci soforu cagirdi")


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb, cls.test = load_kb(), read_jsonl(TEST_PATH)

    def test_kb_shape(self):
        for label, info in self.kb.items():
            with self.subTest(label=label):
                self.assertGreaterEqual(len(info["belirtiler"]), 5)
                self.assertTrue(info.get("anahtar_kelimeler"), "anahtar_kelimeler eksik")
                self.assertIn(info["oncelik"], ("düşük", "orta", "yüksek"))
                self.assertTrue(info["oneri"])
                self.assertEqual(len(info["belirtiler"]), len(set(info["belirtiler"])))

    def test_test_labels_exist_in_kb(self):
        self.assertFalse({r["label"] for r in self.test} - set(self.kb))

    def test_every_class_has_test_sentences(self):
        self.assertFalse(set(self.kb) - {r["label"] for r in self.test})

    def test_no_duplicate_test_sentences(self):
        keys = [tr_lower(r["text"]) for r in self.test]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_leak_into_training_files(self):
        banned = {tr_lower(r["text"]) for r in self.test}
        for name in TRAIN_FILES:
            leaked = [r["text"] for r in read_jsonl(DATA / f"{name}.jsonl") if tr_lower(r["text"]) in banned]
            with self.subTest(file=name):
                self.assertEqual(leaked, [])


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chat import load_model
        cls.model, cls.kb, cls.test = load_model(), load_kb(), read_jsonl(TEST_PATH)

    def test_classes_match_kb(self):
        self.assertEqual(set(self.model.classes_), set(self.kb))

    def test_probabilities_sum_to_one(self):
        from chat import predict
        self.assertAlmostEqual(sum(p for _, p in predict(self.model, "frenlerim ses yapıyor")), 1.0, places=5)

    def test_accuracy_floor(self):
        pred = self.model.predict([r["text"] for r in self.test])
        acc = sum(a == r["label"] for a, r in zip(pred, self.test)) / len(self.test)
        self.assertGreaterEqual(acc, 0.75, f"doğruluk {acc:.3f} tabanın altına düştü")

    def test_clear_cases(self):
        from chat import predict
        cases = {"Fren pedalına basınca metal sürtme sesi geliyor": "fren",
                 "Radyatör fanı çalışmıyor motor kaynıyor": "sogutma",
                 "Triger kayışı değişmesi gerekiyor": "triger"}
        for text, gold in cases.items():
            with self.subTest(text=text):
                self.assertEqual(predict(self.model, text)[0][0], gold)

    def test_ascii_robustness(self):
        same = sum(self.model.predict([r["text"]])[0] == self.model.predict([strip_diacritics(r["text"])])[0]
                   for r in self.test) / len(self.test)
        self.assertGreaterEqual(same, 0.85)


class TestFollowup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chat import ask_followup, load_model, predict
        cls.model, cls.kb = load_model(), load_kb()
        cls.ask, cls.predict = staticmethod(ask_followup), staticmethod(predict)

    def test_returns_symptom_of_top_two(self):
        ranked = self.predict(self.model, "aracım garip davranıyor")
        s = self.ask(self.kb, ranked, set(), self.model)
        self.assertIn(s, self.kb[ranked[0][0]]["belirtiler"] + self.kb[ranked[1][0]]["belirtiler"])

    def test_never_repeats_and_ends(self):
        ranked = self.predict(self.model, "aracım garip davranıyor")
        asked = set()
        for _ in range(40):
            s = self.ask(self.kb, ranked, asked, self.model)
            if s is None:
                break
            self.assertNotIn(s, asked)
            asked.add(s)
        self.assertIsNone(self.ask(self.kb, ranked, asked, self.model))


def cmd_test(_args=None):
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ====================================================================== giriş

def main():
    ap = argparse.ArgumentParser(description="Arıza botu araç kutusu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    p = sub.add_parser("degerlendir")
    p.add_argument("--file")
    p.add_argument("--chat", action="store_true")
    p.set_defaults(fn=cmd_degerlendir)
    p = sub.add_parser("ekle")
    p.add_argument("--file")
    p.set_defaults(fn=cmd_ekle)
    sub.add_parser("durum").set_defaults(fn=cmd_durum)
    sub.add_parser("agirlik").set_defaults(fn=cmd_agirlik)
    sub.add_parser("denetle").set_defaults(fn=cmd_denetle)
    sub.add_parser("yama").set_defaults(fn=cmd_yama)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
