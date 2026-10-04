"""Eğitim verisi üretir.

  python generate_data.py --mode llm        # Claude API ile (ANTHROPIC_API_KEY gerekir)
  python generate_data.py --mode template   # İnternetsiz, şablonlarla (varsayılan)

Üretilen cümleler data/synthetic.jsonl dosyasına yazılır. Test setindeki
cümleler eğitim verisine asla alınmaz (sızıntı olmasın diye).
"""
import argparse
import json
import os
import random

from common import DATA, augment, load_kb, read_jsonl, tr_lower, write_jsonl

PREFIX = ["", "", "", "Merhaba, ", "Arkadaşlar ", "Bugün ", "Son zamanlarda ", "Birkaç gündür ",
          "İyi günler, ", "Usta bey ", "Hocam ", "Dün akşamdan beri ", "Geçen haftadan beri ", "Servise gitmeden önce sorayım, "]
SUFFIX = ["", "", ".", " ne olabilir?", " sizce sorun ne?", " ve bu giderek kötüleşiyor.",
          " ama servise götürmedim.", " hafta başından beri.", " yardımcı olur musunuz?", " ne yapmalıyım?",
          " bir fikri olan var mı?", " çok rahatsız edici."]
SUBJECT = ["aracımda", "arabamda", "", "", "bizim araçta", "otomobilimde", "arabada", "dizel aracımda", "2015 model aracımda"]
CONTEXT = ["", "", "soğuk havada ", "yokuş çıkarken ", "sabahları ", "uzun yolda ",
           "şehir içinde ", "yağmurda ", "hızlanırken ", "dururken ", "trafikte ", "yaz sıcağında ",
           "otoyolda ", "ilk çalıştırmada ", "akşamları "]


def template_sentences(kb: dict, per_class: int, rng: random.Random) -> list[dict]:
    rows = []
    for label, info in kb.items():
        seen = set()
        tries = 0
        while len(seen) < per_class and tries < per_class * 30:
            tries += 1
            k = rng.choice([1, 1, 2])
            parts = rng.sample(info["belirtiler"], k)
            body = " ve ".join(parts) if k == 2 else parts[0]
            text = (rng.choice(PREFIX) + rng.choice(SUBJECT) + " " + rng.choice(CONTEXT)
                    + body + rng.choice(SUFFIX)).strip()
            text = " ".join(text.split())
            seen.add(text[0].upper() + text[1:])
        rows += [{"text": t, "label": label} for t in seen]
    return rows


LLM_PROMPT = """Bir araç sahibinin servise yazdığı mesajları taklit et.
Arıza: {ad}
Bu arızanın tipik belirtileri: {belirtiler}

{n} adet FARKLI Türkçe kullanıcı mesajı yaz. Kurallar:
- Teknik terim bilmeyen sıradan sürücü gibi yaz; bazıları kısa, bazıları uzun olsun.
- Bazılarında günlük konuşma dili, yazım hatası ya da eksik noktalama olsun.
- Her mesaj yalnızca bu arızayı düşündürsün, arızanın adını doğrudan söyleme.
- Yukarıdaki belirti cümlelerini birebir kopyalama, kendi sözcüklerinle anlat.
Yalnızca JSON dizisi döndür: ["mesaj1", "mesaj2", ...]"""


def llm_sentences(kb: dict, per_class: int) -> list[dict]:
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic  (ve ANTHROPIC_API_KEY ayarla)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY ortam değişkeni yok.")
    client = anthropic.Anthropic()
    model = os.environ.get("ARIZA_LLM_MODEL", "claude-sonnet-5-5")
    rows = []
    for label, info in kb.items():
        got: list[str] = []
        while len(got) < per_class:
            n = min(40, per_class - len(got))
            msg = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": LLM_PROMPT.format(
                    ad=info["ad"], belirtiler="; ".join(info["belirtiler"]), n=n)}],
            )
            text = msg.content[0].text
            try:
                batch = json.loads(text[text.index("["): text.rindex("]") + 1])
            except ValueError:
                continue
            got += [b.strip() for b in batch if isinstance(b, str) and b.strip()]
        print(f"{label}: {len(got)} cümle")
        rows += [{"text": t, "label": label} for t in got[:per_class]]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--augment", type=float, default=0.5,
                    help="her cümle için varyant (Türkçe karaktersiz, yazım hatalı...) oranı")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    kb = load_kb()
    rng = random.Random(args.seed)
    rows = (llm_sentences(kb, args.per_class) if args.mode == "llm"
            else template_sentences(kb, args.per_class, rng))

    extra = [{"text": augment(r["text"], rng), "label": r["label"]}
             for r in rows if rng.random() < args.augment]
    rows += extra

    banned = {tr_lower(r["text"]) for r in read_jsonl(DATA / "test_real.jsonl")}
    unique, seen = [], set()
    for r in rows:
        key = tr_lower(r["text"])
        if key in banned or key in seen:
            continue
        seen.add(key)
        unique.append(r)
    write_jsonl(DATA / "synthetic.jsonl", unique)
    print(f"{len(unique)} cümle yazıldı -> data/synthetic.jsonl ({args.mode})")


if __name__ == "__main__":
    main()
