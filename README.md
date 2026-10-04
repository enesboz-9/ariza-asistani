# Araç Arıza Asistanı (prototip)

Kullanıcının anlattığı belirtiden olası arızayı tahmin eden küçük bir sohbet botu.
Karar küçük bir ML modelinden gelir (~0.1 MB); LLM yalnızca eğitim verisi üretmek için kullanılır.

## Kurulum
    pip install -r requirements.txt

## Kullanım
    python run_pipeline.py            # çevrimdışı şablon verisiyle eğit
    python run_pipeline.py --llm      # Claude ile veri üret (ANTHROPIC_API_KEY gerekir)
    python chat.py                    # sohbeti başlat

## Daha büyük veri
- `data/knowledge_base.json` içindeki `anahtar_kelimeler` (parça adları: buji, radyatör, turbo hortumu...) şablon üretimine "buji bozuk galiba" tipi cümleler ekler. Gerçek sürücüler belirti kadar parça adı da söyler.
- Bot, güvensiz olduğunda iki arızayı en iyi ayıran belirtiyi sorar (eskiden listedeki ilkini soruyordu).
- `generate_data.py --per-class 300` (varsayılan) şablonlardan ~8000 cümle üretir; her cümlenin %50'si için
  Türkçe karaktersiz / yazım hatalı / noktalamasız varyant eklenir.
- `python run_pipeline.py --llm --per-class 500` Claude'a çok çeşitli kullanıcı cümleleri yazdırır.
  Şablon verisi yalnızca bilgi tabanındaki ifadeleri bildiği için gerçek dile genellemesi zayıftır; asıl kazanç buradan gelir.
- Kendi CSV veri setin için: `python import_csv.py veri.csv --text-col ... --label-col ... --map eslesme.json`
  (etiketler `data/knowledge_base.json` anahtarlarına eşlenir, sonuç `data/external.jsonl` dosyasına eklenir).

## Kendi kendine gelişim (öğretmen-öğrenci, Groq)
    set GROQ_API_KEY=...           # console.groq.com'dan ücretsiz anahtar
    python self_play.py            # çalışır; kota bitince bekler, yenilenince kaldığı yerden devam eder
    python self_play.py --mock     # API'siz akış denemesi
- Simüle sürücü (LLM) gizli bir arıza seçip belirtiyi anlatır, botun sorularına cevap verir.
- Bot yanılırsa öğretmen (LLM) o arıza için 5 zor anlatım yazar; hepsi `data/selfplay.jsonl` dosyasına gider.
- Bot'un çok yanıldığı arızalar daha sık seçilir. Her 25 konuşmada yeniden eğitim yapılır;
  yeni model, şimdiye kadarki en iyi skordan 0.01'den fazla kötüyse kaydedilmez (84 cümlelik testte daha küçük farklar gürültüdür). Öz-oyun, geri bildirim ve CSV cümleleri eğitimde şablon cümlelerden 3-5 kat ağırlıklıdır; yoksa 8000 şablon cümle birkaç yeni cümleyi bastırır.
- Zor mod artık varsayılan: sürücü belirtiyi belirsiz ve kısa anlatır. `--easy` yalnızca akış denemesi içindir; kolay modda %100 doğruluk bir şey ölçmez (sürücü bilgi tabanındaki cümleleri neredeyse aynen tekrarlar).
- Seçilen modeller `selfplay_state.json` içinde saklanır, her açılışta yeniden denenmez.
- Model başka bir scikit-learn sürümüyle kaydedilmişse program bunu fark edip bu bilgisayarda yeniden eğitir.
- Günlük kota dolunca `data/selfplay_state.json` içine bekleme süresi yazılır; uygulama kapansa bile
  yeniden başlatınca bekleme biter, döngü devam eder. Konuşma kayıtları: `data/selfplay_log.jsonl`.
- Modeller otomatik seçilir: hesabında çalışan ilk aday kullanılır (llama-3.x ücretsiz hesapta kapalı olabilir,
  o zaman openai/gpt-oss-20b / 120b denenir). Elle seçmek için `SIM_MODEL`, `TEACHER_MODEL` ortam değişkenleri;
  hesabındaki modeller için `python self_play.py --list-models`.

## Akış
1. `generate_data.py`  → `data/synthetic.jsonl` (LLM veya şablon)
2. `train.py`          → TF-IDF + lojistik regresyon; **elle yazılmış** `data/test_real.jsonl` ile ölçülür.
   Yeni model eskisinden kötüyse kaydedilmez.
3. `chat.py`           → model emin değilse soru sorar; geri bildirimi `data/feedback_pending.jsonl` dosyasına yazar.
4. Geri bildirimi **sen kontrol edip** doğru satırları `label` alanıyla `data/feedback_verified.jsonl` dosyasına
   taşırsın (`{"text": "...", "label": "fren"}`), sonra `python train.py`.

## Sınırlar
- Test seti küçük (84 cümle); `train.py` doğruluğun bootstrap aralığını yazar (şu an ~%81-94). Şablon verisiyle ölçülen doğruluk yaklaşık %88'dir; bilgi tabanındaki ifadelerden uzak anlatımlarda ve belirtileri birbirine yakın arızalarda (akü / alternatör / marş, lastik / direksiyon) hata yapar. Gerçek kullanıcı cümleleriyle test setini büyüt.
- Sentetik veri gerçek dilden farklıdır; geri bildirim verisi geldikçe iyileşir.
- Bilgi tabanı (`data/knowledge_base.json`) örnek amaçlıdır, bir uzmana doğrulat.
- Ön tahmindir, kesin teşhis değildir.
