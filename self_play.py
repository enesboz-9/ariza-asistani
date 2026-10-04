"""Öğretmen-öğrenci döngüsü: botu LLM ile konuşturup kendi kendine geliştirir.

  set GROQ_API_KEY=...                      (Windows)   export GROQ_API_KEY=...  (Linux/Mac)
  python self_play.py                       # kota bitene kadar, sonra bekler, yenilenince devam eder
  python self_play.py --max-episodes 200    # 200 konuşmada dur
  python self_play.py --mock                # API'siz akış denemesi

Rolleri:
  Simüle sürücü (LLM)  gizli bir arıza seçer, belirtiyi kendi sözleriyle anlatır, botun sorularına cevap verir.
  Bot (senin modelin)  sorar ve tahmin eder (chat.py ile aynı mantık).
  Öğretmen (LLM)       bot yanılırsa, o arıza için zor ve farklı anlatımlar yazar.
Doğru cevap baştan bilindiği için etiketler güvenilirdir. Ölçüm hâlâ elle yazılmış
data/test_real.jsonl üzerinde yapılır; yeni model eskisinden kötüyse kaydedilmez (train.py).

Kota: dakika limitinde kısa bekler. Günlük kota bitince durumu kaydeder, sıfırlanma
süresi kadar uyur ve kaldığı yerden devam eder. Ctrl+C ile durdurup sonra tekrar başlatabilirsin.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import joblib

from chat import CONF, MARGIN, MAX_QUESTIONS, ask_followup, load_model, predict
from common import DATA, MODELS, load_kb, read_jsonl, tr_lower
from llm import GroqClient, MockClient, ModelUnavailable, RateLimit

# Windows konsol/yönlendirme kodlaması ne olursa olsun Türkçe ve simgeler çökmesin
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STATE_PATH = DATA / "selfplay_state.json"
LOG_PATH = DATA / "selfplay_log.jsonl"
OUT_PATH = DATA / "selfplay.jsonl"
SHORT_WAIT = 180          # bu kadar saniyeden kısa beklemeler "dakika limiti" sayılır


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- model seçimi

# Hesapta çalışan ilk modeli kullanır. Ücretsiz hesaplarda llama-3.x "enterprise" olabilir.
SIM_CANDIDATES = ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
TEACHER_CANDIDATES = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def pick_model(client, state, env_name: str, candidates: list[str], role: str) -> str:
    """Ortam değişkeni verilmişse onu, yoksa adaylardan çalışan ilkini seçer (kısa deneme çağrısıyla)."""
    forced = os.environ.get(env_name)
    cached = state.get("models", {}).get(env_name)
    if cached and not forced and cached in candidates:
        return cached            # her açılışta boşuna deneme çağrısı yapma
    for name in ([forced] if forced else candidates):
        try:
            with_quota(lambda: client.chat(name, [{"role": "user", "content": "Merhaba, tek kelime cevap ver."}],
                                           max_tokens=20), state)
            state.setdefault("models", {})[env_name] = name
            save_state(state)
            return name
        except ModelUnavailable:
            log(f"{role}: '{name}' bu hesapta yok/kapalı, sıradaki deneniyor...")
    sys.exit(f"{role} için çalışan model bulunamadı. 'python self_play.py --list-models' ile hesabındaki "
             f"modelleri gör, sonra {env_name} ortam değişkenine yaz.")


# ---------------------------------------------------------------- durum

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"episodes": 0, "stats": {}, "paused_until": None, "since_retrain": 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def pick_fault(kb: dict, state: dict, rng: random.Random) -> str:
    """Bot'un daha çok yanıldığı arızaları daha sık seç."""
    labels = list(kb)
    weights = []
    for label in labels:
        s = state["stats"].get(label, {"n": 0, "wrong": 0})
        err = (s["wrong"] + 1) / (s["n"] + 2)      # yumuşatılmış hata oranı
        weights.append(0.3 + err)
    return rng.choices(labels, weights)[0]


# ---------------------------------------------------------------- kota

def with_quota(call, state: dict):
    """LLM çağrısını yapar; kota hatasında uyur ve aynı çağrıyı yeniden dener."""
    while True:
        try:
            return call()
        except RateLimit as e:
            if e.daily or e.wait > SHORT_WAIT:
                until = datetime.now() + timedelta(seconds=e.wait)
                state["paused_until"] = until.isoformat(timespec="seconds")
                save_state(state)
                log(f"Günlük kota doldu. {until:%d.%m %H:%M} saatine kadar bekleniyor "
                    f"(~{e.wait / 60:.0f} dk). Sonra otomatik devam eder.")
            else:
                log(f"Dakika limiti, {e.wait:.0f} sn bekleniyor...")
            sleep_with_heartbeat(e.wait + 2)
            state["paused_until"] = None
            save_state(state)


def sleep_with_heartbeat(seconds: float) -> None:
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 60))


# ---------------------------------------------------------------- rol komutları

SIM_FIRST = """Sen aracı bozulmuş sıradan bir sürücüsün. Teknik terim bilmezsin.
Aracındaki sorunu servise yazar gibi, Türkçe ve gündelik dille, 1-2 cümleyle anlat.
Sorunun adını söyleme. Belirtiler: {belirtiler}
Bu belirtilerden bir ya da ikisini kendi sözlerinle anlat; cümleleri kopyalama.{zorluk}
Yalnızca mesajı yaz."""

HARD_EXTRA = (" Belirsiz ve kısa anlat, tek bir belirtiyi söyle, listedeki kelimeleri kullanma,"
              " günlük deyimler ve eksik bilgi kullan, bazen yanlış bir tahminini de ekle.")
difficulty = {"extra": ""}

SIM_ANSWER = """Sen aracı bozulmuş sıradan bir sürücüsün. Servis sana şunu sordu: "{soru}"
Doğru cevap: {cevap}
Bu cevabı kısa (en fazla bir cümle), gündelik Türkçe ile söyle. {ek}
Belirtiler: {belirtiler}
Yalnızca cevabı yaz."""

TEACHER = """Bir araç arıza botu sürücü mesajını yanlış sınıflandırdı.
Doğru arıza: {dogru}
Botun yanlış tahmini: {yanlis}
Botu yanıltan mesaj: "{mesaj}"
Arıza belirtileri: {belirtiler}
Aynı doğru arıza için, botu zorlayacak 5 FARKLI sürücü mesajı yaz: günlük dil, bazıları kısa,
bazılarında yazım hatası, arızanın adını söyleme, belirtileri birebir kopyalama.
Yalnızca JSON dizisi döndür: ["mesaj1", "mesaj2", ...]"""


# ---------------------------------------------------------------- bir konuşma

def run_episode(client, kb, model, state, rng, sim_model, teacher_model):
    label = pick_fault(kb, state, rng)
    info = kb[label]
    sym_text = "; ".join(info["belirtiler"])
    text = with_quota(lambda: client.chat(
        sim_model, [{"role": "user", "content": SIM_FIRST.format(belirtiler=sym_text, zorluk=difficulty["extra"])}]), state)
    transcript = [("sürücü", text)]

    ranked = predict(model, text)
    asked, questions = set(), 0
    while (ranked[0][1] < CONF or ranked[0][1] - ranked[1][1] < MARGIN) and questions < MAX_QUESTIONS:
        symptom = ask_followup(kb, ranked, asked, model)
        if symptom is None:
            break
        asked.add(symptom)
        questions += 1
        yes = symptom in info["belirtiler"]       # doğru cevap bilinen veriden gelir
        answer = with_quota(lambda: client.chat(sim_model, [{"role": "user", "content": SIM_ANSWER.format(
            soru=symptom + "?", cevap="EVET" if yes else "HAYIR",
            ek="Evet ise kısaca ayrıntı ver." if yes else "", belirtiler=sym_text)}]), state)
        transcript += [("bot", symptom + "?"), ("sürücü", answer)]
        if yes:
            text += ". " + answer
            ranked = predict(model, text)

    predicted = ranked[0][0]
    correct = predicted == label
    s = state["stats"].setdefault(label, {"n": 0, "wrong": 0})
    s["n"] += 1
    s["wrong"] += 0 if correct else 1

    new_rows = [{"text": transcript[0][1], "label": label}]       # ilk mesaj her zaman öğrenilir
    if not correct:
        raw = with_quota(lambda: client.chat(teacher_model, [{"role": "user", "content": TEACHER.format(
            dogru=info["ad"], yanlis=kb[predicted]["ad"], mesaj=text, belirtiler=sym_text)}],
            temperature=1.0, max_tokens=700), state)
        try:
            variants = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            new_rows += [{"text": v.strip(), "label": label}
                         for v in variants if isinstance(v, str) and v.strip()]
        except ValueError:
            log("Öğretmen cevabı JSON değil, varyantlar atlandı.")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "dogru": label,
                            "tahmin": predicted, "dogru_mu": correct, "soru_sayisi": questions,
                            "konusma": transcript}, ensure_ascii=False) + "\n")
    return correct, label, predicted, new_rows


def append_rows(rows: list[dict]) -> int:
    """Tekrarları ve test cümlelerini eleyip selfplay.jsonl'e ekler."""
    banned = {tr_lower(r["text"]) for r in read_jsonl(DATA / "test_real.jsonl")}
    existing = read_jsonl(OUT_PATH)
    seen = {tr_lower(r["text"]) for r in existing}
    added = 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for r in rows:
            k = tr_lower(r["text"])
            if k in banned or k in seen:
                continue
            seen.add(k)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    return added


def retrain() -> None:
    log("Yeniden eğitim (kapı: elle yazılmış test seti)...")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}   # Windows kodlama sorunu
    out = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in (out.stdout or "").splitlines():
        if any(k in line for k in ("Doğruluk", "kaydedildi", "korundu", "kötü", "düşüş", "karışan")):
            log("  " + line.strip())
    if out.returncode != 0:
        log("  eğitim hatası: " + (out.stderr or "")[-300:])


# ---------------------------------------------------------------- ana döngü

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = sınırsız")
    ap.add_argument("--retrain-every", type=int, default=25)
    ap.add_argument("--mock", action="store_true", help="API'siz sahte LLM")
    ap.add_argument("--mock-quota-every", type=int, default=0, help="sahte kota hatası sıklığı (deneme)")
    ap.add_argument("--mock-quota-wait", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--easy", action="store_true",
                    help="kolay mod: sürücü belirtileri neredeyse aynen anlatır (akış denemesi için). "
                         "Varsayılan artık zor mod; kolay modda %%100 doğruluk bir şey ölçmez.")
    ap.add_argument("--hard", action="store_true", help=argparse.SUPPRESS)   # eski komutlar bozulmasın
    ap.add_argument("--list-models", action="store_true", help="hesabındaki modelleri listele ve çık")
    args = ap.parse_args()

    if args.list_models:
        for m in GroqClient().list_models():
            print(m)
        return
    if not (MODELS / "ariza_model.joblib").exists():
        sys.exit("Önce temel modeli eğit: python run_pipeline.py")
    if not args.easy:
        difficulty["extra"] = HARD_EXTRA
    kb = load_kb()
    rng = random.Random(args.seed)
    client = (MockClient(args.mock_quota_every, args.mock_quota_wait) if args.mock else GroqClient())
    state = load_state()
    sim_model = pick_model(client, state, "SIM_MODEL", SIM_CANDIDATES, "Simüle sürücü")
    teacher_model = pick_model(client, state, "TEACHER_MODEL", TEACHER_CANDIDATES, "Öğretmen")

    # önceki oturumda kota beklemesi yarım kaldıysa bitirene kadar uyu
    if state.get("paused_until"):
        left = (datetime.fromisoformat(state["paused_until"]) - datetime.now()).total_seconds()
        if left > 0:
            log(f"Önceki kota beklemesi sürüyor, {left / 60:.0f} dk uyunuyor...")
            sleep_with_heartbeat(left + 2)
        state["paused_until"] = None

    model = load_model()
    done = correct_total = 0
    log(f"Başlıyor. Simüle sürücü: {sim_model} | öğretmen: {teacher_model} | toplam konuşma: {state['episodes']}")
    try:
        while not args.max_episodes or done < args.max_episodes:
            ok, label, pred, rows = run_episode(client, kb, model, state, rng, sim_model, teacher_model)
            added = append_rows(rows)
            done += 1
            correct_total += ok
            state["episodes"] += 1
            state["since_retrain"] += 1
            save_state(state)
            log(f"#{state['episodes']} {'✓' if ok else '✗'} gerçek={label} tahmin={pred} "
                f"(+{added} yeni cümle, bu oturum doğruluk %{100 * correct_total / done:.0f})")
            if state["since_retrain"] >= args.retrain_every:
                retrain()
                model = load_model()
                state["since_retrain"] = 0
                save_state(state)
    except KeyboardInterrupt:
        log("Durduruldu. Durum kaydedildi; tekrar çalıştırınca devam eder.")
    if state["since_retrain"] > 0:
        retrain()
    if done:
        log(f"Oturum bitti: {done} konuşma, doğruluk %{100 * correct_total / done:.0f}, "
            f"kullanılan token ~{client.tokens_used}")


if __name__ == "__main__":
    main()
