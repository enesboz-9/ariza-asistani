"""LLM istemcileri: Groq (ücretsiz katman) ve çevrimdışı sahte istemci.

Groq API'si OpenAI uyumludur. Anahtar:  GROQ_API_KEY  (console.groq.com)
Model adları zamanla değişebilir; ortam değişkeniyle ayarlanır:
  SIM_MODEL      simüle sürücü (çok çağrı, küçük/hızlı model)   varsayılan llama-3.1-8b-instant
  TEACHER_MODEL  öğretmen (yalnızca hatalarda)                 varsayılan llama-3.3-70b-versatile
"""
import json
import os
import random
import re
import urllib.error
import urllib.request

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class ModelUnavailable(Exception):
    """Model yok ya da bu hesapta kapalı (HTTP 404 / model_not_found)."""


class RateLimit(Exception):
    """Kota doldu. wait: kaç saniye sonra tekrar denenmeli. daily: günlük kota mı?"""

    def __init__(self, wait: float, daily: bool, message: str = ""):
        super().__init__(message)
        self.wait = wait
        self.daily = daily


def _parse_wait(headers, body: str) -> float:
    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    # "Please try again in 1h2m3.4s" biçimi
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", body)
    if m and any(m.groups()):
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    return 600.0


class GroqClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise SystemExit("GROQ_API_KEY ortam değişkeni yok (console.groq.com'dan ücretsiz anahtar al).")
        self.tokens_used = 0

    def chat(self, model: str, messages: list[dict], temperature: float = 0.9,
             max_tokens: int = 400) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if model.startswith("openai/gpt-oss"):
            # Akıl yürütme modelleri: düşünme tokenları max_tokens'ı yer, cevap boş kalabilir.
            body["reasoning_effort"] = "low"
            body["max_tokens"] = max(max_tokens, 1200)
        payload = json.dumps(body).encode()
        req = urllib.request.Request(GROQ_URL, data=payload, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ariza-bot/1.0",   # varsayılan python UA bazı ağlarda engelleniyor
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 429:
                wait = _parse_wait(e.headers, body)
                daily = "per day" in body.lower() or "TPD" in body or "RPD" in body or wait > 180
                raise RateLimit(wait, daily, body[:200]) from None
            if e.code in (400, 403, 404) and ("model" in body.lower()):
                raise ModelUnavailable(f"{model}: {body[:200]}") from None
            raise RuntimeError(f"Groq HTTP {e.code}: {body[:300]}") from None
        self.tokens_used += data.get("usage", {}).get("total_tokens", 0)
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise RuntimeError(f"{model} boş cevap döndürdü (akıl yürütme modeli token'ı bitirmiş olabilir).")
        return content

    def list_models(self) -> list[str]:
        req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers={
            "Authorization": f"Bearer {self.api_key}", "User-Agent": "ariza-bot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return sorted(m["id"] for m in json.load(resp)["data"])


class MockClient:
    """API'siz deneme için: gerçekçi olmayan ama akışı sınayan sahte LLM.

    quota_every=N: her N çağrıda sahte 'günlük kota doldu' hatası verir (bekleme: quota_wait sn).
    """

    def __init__(self, quota_every: int = 0, quota_wait: float = 2.0, seed: int = 1):
        self.calls = 0
        self.quota_every = quota_every
        self.quota_wait = quota_wait
        self.tokens_used = 0
        self.rng = random.Random(seed)
        self._blocked_until_call = 0

    def chat(self, model: str, messages: list[dict], temperature: float = 0.9,
             max_tokens: int = 400) -> str:
        if self.quota_every and (self.calls + 1) % self.quota_every == 0 \
                and self.calls != self._blocked_until_call:
            self._blocked_until_call = self.calls   # bir sonraki deneme geçsin
            raise RateLimit(self.quota_wait, True, "sahte günlük kota")
        self.calls += 1
        self.tokens_used += 300
        prompt = messages[-1]["content"]
        if "JSON dizisi" in prompt:     # öğretmen: varyant istiyor
            m = re.search(r"Arıza belirtileri: (.*)", prompt)
            symptoms = [s.strip() for s in (m.group(1) if m else "araç bozuk").split(";")]
            return json.dumps([f"{self.rng.choice(['Hocam', 'Merhaba', 'Usta'])} {self.rng.choice(symptoms)} galiba"
                               for _ in range(5)], ensure_ascii=False)
        m = re.search(r"Belirtiler: (.*)", prompt)
        symptoms = [s.strip() for s in (m.group(1) if m else "bir sorun var").split(";")]
        if "EVET" in prompt:
            return f"Evet, {self.rng.choice(symptoms)} aynen"
        if "HAYIR" in prompt:
            return "Hayır, öyle bir şey yok"
        return f"{self.rng.choice(symptoms)}, biraz da {self.rng.choice(symptoms)}"
