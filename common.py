"""Ortak yardımcılar."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def tr_lower(text: str) -> str:
    """Türkçe'ye uygun küçük harf (I -> ı, İ -> i)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def load_kb() -> dict:
    return json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


_DIACRITICS = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def strip_diacritics(text: str) -> str:
    """'öğrenci' -> 'ogrenci' (telefonda Türkçe karakter kullanmayanlar için)."""
    return text.translate(_DIACRITICS)


def add_typo(text: str, rng) -> str:
    """Rastgele bir yerde iki harfin yerini değiştirir."""
    if len(text) < 6:
        return text
    i = rng.randrange(1, len(text) - 2)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def augment(text: str, rng) -> str:
    """Gerçek yazım alışkanlıklarını taklit eden bir varyant üretir."""
    choice = rng.choice(["ascii", "ascii", "typo", "lower", "nopunct"])
    if choice == "ascii":
        return strip_diacritics(text)
    if choice == "typo":
        return add_typo(text, rng)
    if choice == "lower":
        return tr_lower(text)
    return text.replace(".", "").replace("?", "").replace(",", "")
