"""Kendi indirdiğin CSV veri setini eğitime ekler.

  python import_csv.py veri.csv --text-col sikayet --label-col ariza
  python import_csv.py veri.csv --text-col sikayet --label-col ariza --map eslesme.json

--map: veri setindeki etiketleri bu projedeki arıza kodlarına çevirir, örneğin
       {"Brake failure": "fren", "Battery": "aku"}. Eşleşmeyen satırlar atlanır.
Proje arıza kodları: data/knowledge_base.json içindeki anahtarlar.
Sonuç data/external.jsonl dosyasına EKLENİR; tekrar eden cümleler atılır.
Test setindeki cümleler asla alınmaz.
"""
import argparse
import csv
import json
import sys

from common import DATA, load_kb, read_jsonl, tr_lower, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--map", help="etiket eşleme JSON dosyası")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--encoding", default="utf-8-sig")
    args = ap.parse_args()

    kb = load_kb()
    mapping = json.load(open(args.map, encoding="utf-8")) if args.map else None
    banned = {tr_lower(r["text"]) for r in read_jsonl(DATA / "test_real.jsonl")}
    rows = read_jsonl(DATA / "external.jsonl")
    seen = {tr_lower(r["text"]) for r in rows}

    added = skipped = 0
    with open(args.csv_path, encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=args.delimiter)
        for col in (args.text_col, args.label_col):
            if col not in (reader.fieldnames or []):
                sys.exit(f"Sütun yok: {col}. Mevcut sütunlar: {reader.fieldnames}")
        for rec in reader:
            text = (rec[args.text_col] or "").strip()
            label = (rec[args.label_col] or "").strip()
            label = mapping.get(label) if mapping else label
            key = tr_lower(text)
            if not text or label not in kb or key in banned or key in seen:
                skipped += 1
                continue
            seen.add(key)
            rows.append({"text": text, "label": label})
            added += 1
    write_jsonl(DATA / "external.jsonl", rows)
    print(f"{added} cümle eklendi, {skipped} satır atlandı. Toplam dış veri: {len(rows)}")


if __name__ == "__main__":
    main()
