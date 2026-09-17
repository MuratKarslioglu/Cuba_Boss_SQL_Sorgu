# Turkish Text → SQL-JSON Extractor

Türkçe doğal dil isteklerini, sonradan SQL üretmek için kullanılacak **katı şemalı bir JSON** temsiline çeviren küçük, yerel ve deterministik bir sistem.

```text
Türkçe Doğal Dil
        ↓
Küçük Fine-Tuned Dil Modeli
        ↓
Katı Sorgu JSON'u
        ↓
Şema Doğrulaması
```

Model **serbest SQL üretmez**. Görevi yalnızca Türkçe bir isteği önceden tanımlı bir JSON nesnesine çevirmektir. SQL üretimi ileride deterministik bir builder'a bırakılacaktır — bu tasarım halüsinasyon, SQL sözdizimi hatası ve güvenlik riskini önemli ölçüde azaltır.

Tam tasarım dokümanı: [`docs/text_to_sql_json_project_spec.md`](docs/text_to_sql_json_project_spec.md)

---

## Durum

| Faz | Kapsam | Durum |
|---|---|---|
| 1 | Repo iskeleti, config, Pydantic şeması, testler | **Tamam** |
| 2 | Dataset loader, doğrulayıcı, eğitim formatlayıcı | **Tamam** |
| 3 | Model yükleyici, cihaz seçimi, LoRA eğitim scripti | **Altyapı tamam** — gerçek prod eğitimi başlatılmadı |
| 4 | Prompt formatlayıcı → inference, JSON parse, CLI | **Altyapı tamam** |
| 5 | Exact match, semantic match, alan bazlı metrikler, benchmark | **Altyapı tamam** |
| 6 | FastAPI `/extract`, `/health` | Yapılmadı |
| 7 | Optimizasyon (benchmark verisine dayalı) | Yapılmadı |

`datasets/text_to_sql_json_dataset_v1_1/` sağlandı (700/150/150, spec §10 ile birebir) ve `scripts/validate_dataset.py` ile bağımsızca doğrulandı (0 hata). Faz 3'ün eğitim altyapısı (`app/inference/model_loader.py`, `training/train.py`) gerçek küçük bir modelle (`hf-internal-testing/tiny-random-gpt2`) uçtan uca test edildi; gerçek üretim modeliyle (0.5B–1.5B sınıfı) tam eğitim çalıştırılmadı.

---

## Mimari

```text
app/
  config.py            Merkezi konfigürasyon (.env). torch import etmez.
  errors.py            Uygulama hata tipleri (spec §32)
  schemas/
    enums.py           QueryOperation, FilterOperator, OrderDirection, QueryStatus (V2)
    query.py           QueryRequest + kanonik JSON serileştirme + V2 status sözleşmesi
    semantics.py       Aynı alanda çelişen filtre tespiti (eq/eq, eq/neq, eq/in, aralık çelişkileri)
    vocabulary.py      Alan/hedef sözlüğü (geçici, sadece doğrulamada)
  normalization/
    text.py            Hafif boşluk normalizasyonu + dedup anahtarı

training/
  dataset.py           JSONL yükleme, tekrar/sızıntı tespiti, dağılım raporu
  formatting.py        Prompt formatlayıcı soyutlaması (plain / chat template)
  train.py             LoRA/SFT eğitim CLI'ı (torch/trl yalnızca çalıştırılınca yüklenir)

app/inference/
  model_loader.py       Cihaz/dtype seçimi, model+tokenizer yükleme, LoRA uygulama
  parsing.py            Model çıktısından JSON izolasyon + Pydantic doğrulama (torch'tan bağımsız)
  extractor.py           Deterministik üretim + tek seferlik onarım orkestrasyonu

training/evaluate.py     Exact/semantic match, alan bazlı doğruluk, gecikme/token istatistikleri (torch'tan bağımsız)

scripts/
  validate_dataset.py  Dataset bütünlük doğrulayıcısı (CLI)
  run_inference.py     Yerel CLI çıkarımı (CLI)
  benchmark.py         Test seti üzerinde performans/doğruluk benchmarki (CLI)

tests/                 pytest paketi — gerçek model yüklemez
datasets/              Dataset dosyaları buraya konur (repoda tutulmaz)
models/                LoRA adapter çıktıları
```

**Sorumluluk ayrımı bilinçlidir:** eğitim mantığı API dosyalarına, prompt formatlama/Pydantic doğrulama/model yükleme tek bir dosyaya konmaz.

---

## JSON Sözleşmesi (V2)

```json
{
  "operation": "sum",
  "target": "sales",
  "filters": [
    { "field": "company", "operator": "eq", "value": "ABC" }
  ],
  "group_by": ["product"],
  "order_by": { "field": "sales", "direction": "desc" },
  "limit": 5,
  "status": "valid",
  "clarification": null
}
```

Anahtar sırası **sabittir**: `operation, target, filters, group_by, order_by, limit, status, clarification`.

- `operation` ∈ `select, count, sum, average, min, max` (veya `null`, aşağıya bakın)
- `operator` ∈ `eq, neq, gt, gte, lt, lte, between, in, contains`
- `direction` ∈ `asc, desc`
- `status` ∈ `valid, insufficient_information, ambiguous, conflicting`

### V2 eklentisi: `status` / `clarification`

Her Türkçe istek doğrudan yapılandırılmış bir sorguya çevrilemez — belirsiz, çelişkili veya eksik bilgili istekler olabilir. `status` bunu **açıkça** işaretler; böylece belirsiz bir istek sessizce (ve yanlışlıkla) geçerli bir sorguya dönüştürülmez:

- `status="valid"` → `operation` ve `target` **zorunlu**, `clarification` **null olmalı**
- `status≠"valid"` → `operation` ve `target` **null olmalı**, `clarification` **zorunlu** (kullanıcıya gösterilecek kısa açıklama/red gerekçesi)

Bu kural `QueryRequest` üzerinde bir Pydantic doğrulayıcısı olarak uygulanır — model bu iki durumu karıştırırsa (`status="valid"` derken `operation` vermezse, veya `status="ambiguous"` derken tam bir sorgu üretirse) çıktı **VALIDATION_ERROR** ile reddedilir.

**Geriye dönük uyumluluk:** `status`/`clarification` varsayılan değerlere sahiptir (`valid`/`null`), bu alanları hiç içermeyen eski V1 verisi (`datasets/text_to_sql_json_dataset_v1_1/`) değişmeden geçerli kalır.

Doğrulama kuralları (`app/schemas/query.py`):

- `between` → tam 2 elemanlı liste
- `in` → boş olmayan liste
- diğer operatörler → liste **değil**
- `limit` verilmişse pozitif
- **bilinmeyen anahtarlar reddedilir** (`extra="forbid"`) — model uydurma alan üretirse sessizce yutulmaz
- **aynı alanda mantıksal olarak çelişen filtreler reddedilir** (`app/schemas/semantics.py`) — örn. `company eq 'ABC'` ve `company eq 'XYZ'` şemaya uyar ama SQL'e çevrilince hiçbir satırla eşleşmez; bu gerçek bir model çalıştırmasında gözlemlendi ve artık açıkça reddediliyor. Ayrıca sayısal/tarih aralık çelişkileri (`gt 100` + `lt 50`) ve `eq`/`neq`/`in` çelişkileri de yakalanır.

`target` ve `field` bilerek serbest string'tir: bu isimler geçicidir ve işverenin gerçek veritabanı şemasıyla değişecektir. İzin verilen isim listesi `app/schemas/vocabulary.py`'de ayrı tutulur ve yalnızca dataset doğrulamasında uygulanır.

---

## Kurulum

Bu proje **hibrit** bir kurulum kullanır:

- **Eğitim / inference / benchmark → native.** Apple Silicon'da GPU (MPS) yalnızca native ortamdan erişilebilir; Docker Desktop'ın Linux VM'i Metal'e erişemez, dolayısıyla container içinde her şey CPU-only olur ve spec §23'ün latency ölçümleri anlamsızlaşır.
- **Servis / dağıtım → Docker.** FastAPI servisi ve ileride işverenin Linux sunucusuna taşıma bu yoldan gider.

### Native (önerilen geliştirme yolu)

Python **3.12** kullanılır (`.python-version`). Sistem Python'u 3.14 ise torch/transformers tekerlekleri için sorun çıkar.

`uv` ile:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Klasik `venv` ile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

**Yalnızca Faz 1+2 ile çalışacaksanız** (şema, dataset doğrulama, formatlama) torch'a gerek yoktur — ~2.5 GB indirme yerine:

```bash
uv pip install -r requirements-core.txt pytest ruff
```

Ardından `.env` dosyanızı oluşturun:

```bash
cp .env.example .env
```

### Bağımlılık dosyaları

| Dosya | İçerik |
|---|---|
| `requirements-core.txt` | `pydantic`, `pydantic-settings` — Faz 1+2 için yeterli |
| `requirements.txt` | core + `torch`, `transformers`, `peft`, `trl`, `datasets`, `fastapi`, `uvicorn` |
| `requirements-dev.txt` | yukarıdakiler + `pytest`, `ruff` |

`bitsandbytes` yalnızca Linux işaretleyicisiyle listelenir. Apple Silicon'da çalışmaz, bu nedenle **QLoRA opsiyoneldir** ve desteklenmeyen platformda açık bir hatayla başarısız olur (spec §30).

### Docker

```bash
docker build -t text-to-sql-json .
docker run --rm text-to-sql-json          # test paketini çalıştırır
```

Bu imaj **eğitim veya benchmark için kullanılmamalıdır** (GPU erişimi yoktur). Faz 6'da `CMD` uvicorn'a çevrilecek ve `docker-compose.yml` eklenecektir.

---

## Dataset

### Format

JSONL — her satır bağımsız bir denetimli örnek:

```json
{"input":"X firmasındaki Y ürününün satış sayısını getir.","output":{"operation":"count","target":"sales","filters":[{"field":"company","operator":"eq","value":"X"},{"field":"product","operator":"eq","value":"Y"}],"group_by":[],"order_by":null,"limit":null}}
```

Mevcut dataset'ler (repoda tutulmaz, `.gitignore`'dadır):

| Dizin | Boyut | Not |
|---|---|---|
| `datasets/text_to_sql_json_dataset_v1_1/` | 700/150/150 | V1 sözleşmesi (status/clarification yok) |
| `datasets/sql_json_dataset_v2_professional/` | 12000/1977/2973 + 996 challenge | V2 — gerçekçi gürültü (yazım hatası, konuşma dili, telgraf üslubu), `status`/`clarification`, prompt-injection benzeri örnekler |

V2'nin `validation.jsonl`/`test.jsonl`/`challenge_test.jsonl` dosyaları **temizlenmiş** hâlleridir: bağımsız doğrulamamız `train` ile aralarında 54 satırlık büyük/küçük harf-noktalama varyantı sızıntı buldu (üreticinin "global exact-input uniqueness" kontrolü bunu yakalamamıştı — yalnızca birebir string eşitliğine bakıyordu). Sızan satırlar çıkarıldı; orijinaller `*.jsonl.orig` olarak saklanıyor.

Anlamsal olarak neredeyse aynı varyantlar mümkün olduğunca **tek bir split içinde** kalmalıdır — doğrulayıcı hem birebir hem normalize edilmiş (büyük/küçük harf, noktalama farkı) sızıntıyı splitler arası **hata**, aynı split içi normalize tekrarı ise **uyarı** olarak raporlar (organik/büyük dataset'lerde kaçınılmaz, sızıntı değildir).

### Doğrulama

```bash
python scripts/validate_dataset.py \
  --train datasets/sql_json_dataset_v2_professional/train.jsonl \
  --validation datasets/sql_json_dataset_v2_professional/validation.jsonl \
  --test datasets/sql_json_dataset_v2_professional/test.jsonl
```

Kontrol edilenler (spec §21 + V2): her satırın geçerli JSON olması, `input`/`output` varlığı, boş girdi olmaması, `QueryRequest` şema uyumu (status/clarification sözleşmesi dahil), enum geçerliliği, `between`/`in`/`limit`/`direction` kuralları, aynı alanda çelişen filtreler, dosya içi tam tekrar, splitler arası (tam + normalize) sızıntı, **`challenge_test.jsonl`'in yanlışlıkla `--train`/`--validation` olarak verilmesi**.

Sorun bulunursa **0 dışı kod** ile çıkar.

Ek bayraklar:

| Bayrak | Etki |
|---|---|
| `--strict-vocabulary` | Sözlük dışındaki alan/hedef adlarını uyarı yerine hata sayar |
| `--json` | Raporu makine okunabilir JSON olarak yazdırır (CI için) |

Çıktı ayrıca operation / target / operatör / filtre sayısı / **status / family / noise** dağılımını raporlar (V2 alanları yalnızca dataset onları sağlıyorsa görünür).

`challenge_test.jsonl` **asla eğitim/doğrulamada kullanılmamalıdır** — yalnızca benchmark'ta `--test` olarak verin:

```bash
python scripts/validate_dataset.py \
  --train datasets/sql_json_dataset_v2_professional/train.jsonl \
  --test datasets/sql_json_dataset_v2_professional/challenge_test.jsonl
```

---

## Eğitim

```bash
python training/train.py \
  --model_name <HF_MODEL_ID> \
  --train_file datasets/sql_json_dataset_v2_professional/train.jsonl \
  --validation_file datasets/sql_json_dataset_v2_professional/validation.jsonl \
  --output_dir models/run_002 \
  --epochs 3 \
  --learning_rate 2e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

- Model adı ve tüm hiperparametreler CLI'dan gelir; kaynak kodda hiçbir model sabitlenmemiştir (spec §13).
- **`challenge_test.jsonl` üzerinde eğitim otomatik olarak reddedilir** — hem dosya adından hem `metadata.split` alanından kontrol edilir (CLAUDE_V2_MIGRATION.md: "Never train on challenge_test.jsonl").
- Cihaz (`--device auto|cuda|mps|cpu`) ve veri tipi (`--dtype auto|float32|float16|bfloat16`) otomatik seçilir; zorlanan bir cihaz mevcut değilse sessizce başka bir cihaza düşülmez, açık hata verilir (spec §30).
- Prompt formatı (`--prompt_format auto|plain|chat`) varsayılanda tokenizer'ın chat template'i varsa onu, yoksa spec §14'teki düz metin formatını kullanır.
- LoRA uygulanır (tam fine-tuning yapılmaz); veri, TRL'in `prompt`/`completion` kolon formatına çevrilerek yalnızca cevap tokenleri üzerinden loss hesaplanır.
- Çıktı klasörüne kaydedilenler (spec §29): LoRA adapter'i, tokenizer, `training_config.json` (kullanılan tüm hiperparametreler + çözümlenen cihaz/dtype), `eval_summary.json` (doğrulama seti üzerindeki gerçek kayıp/metrikler — asla uydurulmaz, spec §38).
- Determinizm: `--seed` (varsayılan 42) `random`/`numpy`/`torch` tohumlarını sabitler (spec §31). Bu, farklı donanımlarda matematiksel olarak birebir aynı sonucu garanti etmez; sistem sözleşmesi doğrulamaya dayanır.
- QLoRA (4-bit) şu an desteklenmiyor: `bitsandbytes` yalnızca Linux/CUDA'da kurulur, Apple Silicon'da çalışmaz (spec §30).
- Altyapı gerçek bir modelle (`hf-internal-testing/tiny-random-gpt2`) uçtan uca test edilmiştir. Gerçek üretim modeliyle (0.5B–1.5B sınıfı) tam bir eğitim koşusu henüz **başlatılmamıştır** — model seçimi ve koşu, ölçülen sonuçlara dayanacağı için ayrı bir karar gerektirir (spec §13).

## Inference

```bash
python scripts/run_inference.py \
  --text "X firmasındaki Y ürününün satış sayısını getir." \
  --model_name <HF_MODEL_ID_veya_yerel_yol> \
  --adapter_path ./models/latest
```

Örnek çıktı:

```text
Input:
X firmasındaki Y ürününün satış sayısını getir.

Output:
{"operation":"count","target":"sales","filters":[...],"group_by":[],"order_by":null,"limit":null}

Valid: true
Latency: 82 ms
```

- Üretim ayarları tamamen deterministiktir: `do_sample=False`, tek beam (`num_beams=1`). `temperature`/`top_p` hiç geçirilmez (spec §15).
- Model/adapter yolu argümanla verilmezse `.env`'deki `BASE_MODEL`/`ADAPTER_PATH`'e düşer; hiçbir model kaynak kodda sabitlenmemiştir.
- Akış (spec §16): prompt formatla → model çağır → JSON'u metinden izole et (çevresindeki fazladan metne dayanıklı bir parantez-eşleştirici ile, naif `find`/`rfind` değil) → `json.loads` → Pydantic doğrulama → `{"success", "data", "error"}` zarfı.
- Geçersiz çıktı **sessizce onarılmaz**: en fazla bir kez, küçük bir onarım istemiyle yeniden denenir (`--no_retry` ile kapatılabilir — sıfır-retry/tek-retry karşılaştırması için, spec §17); ikinci deneme de başarısız olursa açık bir `JSON_PARSE_ERROR`/`VALIDATION_ERROR` döner.
- `app/inference/parsing.py` torch'tan bağımsızdır ve gerçek model olmadan test edilir; `app/inference/extractor.py` üretim çağrısını sahte (mock) bir modelle test eder (spec §24). Gerçek küçük bir modelle uçtan uca çalışan opsiyonel bir entegrasyon testi de vardır (`RUN_MODEL_INTEGRATION_TESTS=1 pytest`).

## Değerlendirme ve Benchmark

```bash
python scripts/benchmark.py \
  --model_name <HF_MODEL_ID> \
  --adapter_path ./models/run_002 \
  --test_file datasets/sql_json_dataset_v2_professional/test.jsonl \
  --output_dir benchmarks/
```

BLEU/ROUGE **kullanılmaz** (spec §22). Hesaplanan metrikler:

- **Valid JSON oranı** ve **şema geçerlilik oranı** — ayrı ayrı (JSON parse hatası ile şema uyumsuzluğu farklı kategoriler).
- **Strict exact match** — kanonik JSON birebir eşitliği.
- **Semantic structural match** — filtrelerin ve `group_by`'ın *sırası* önemsiz, ama *çokluğu* (kaç kez geçtiği) eşit olmalı (`Counter` ile karşılaştırılır, `set` ile değil). `operation`/`target`/`order_by`/`limit` tam eşit aranır. **Exact match'in yerine geçmez, ikisi de ayrı ayrı raporlanır.**
- **Alan bazlı doğruluk** — `operation`, `target`, filtre `field`/`operator`/`value`, `group_by`, `order_by`, `limit`, **`status`** için ayrı ayrı; geçersiz çıktılarda o örneğin **tüm alanları** yanlış sayılır (payda hep `total_examples` kalır).
- **Rejection precision/recall** (V2) — "bu istek reddedilmeli mi" ikili sınıflandırması. Extraction tamamen başarısız olması (`status` sinyali hiç verilemedi) **"non-valid" ile aynı sayılmaz**: beklenen non-valid iken extraction başarısız olması `false_negative`'dir (kaçırılmış bir red); beklenen valid iken extraction başarısız olması ne `false_positive` ne `true_negative`'dir — ayrı bir `extraction_failed_on_valid` sayacında tutulur.
- **Status confusion matrix** (V2) — beklenen `status` → tahmin edilen `status` (veya extraction başarısızsa `extraction_failed`).
- **`status` / `family` / `noise` bazında kırılım** (V2) — her biri için ayrı bir `EvaluationSummary` (`by_status`, `by_family`, `by_noise`). Dataset bu alanları sağlamıyorsa (V1) boş kalır, hiçbir şey uydurulmaz.
- **Gecikme**: ortalama/medyan/p95 (yalnızca stdlib, numpy'siz doğrusal enterpolasyon).
- **Token sayıları**: ortalama prompt/üretilen token; tekrar denendiyse (spec §17) toplam gecikme ve toplam üretilen token raporlanır.
- **Tepe bellek**: mümkün olduğunda (`resource` modülü, Windows'ta `None`).

Sonuçlar `benchmarks/<model_adı>_<YYYYMMDD>.json`'a yazılır. `--allow_retry` **varsayılan kapalı** (spec §17: ilk-deneme doğruluğu birincil metrik); tek-retry kolunu ölçmek için ayrıca `--allow_retry` ile çalıştırın.

Zorlu/adversarial performansı ölçmek için `challenge_test.jsonl`'i doğrudan `--test_file` olarak verin (eğitimde değil, yalnızca benchmark'ta kullanılmalıdır):

```bash
python scripts/benchmark.py \
  --model_name <HF_MODEL_ID> --adapter_path ./models/run_002 \
  --test_file datasets/sql_json_dataset_v2_professional/challenge_test.jsonl \
  --output_dir benchmarks/
```

`training/evaluate.py` tamamen torch'tan bağımsızdır — `app.inference.parsing.ExtractionResult`i girdi alır, gerçek model olmadan test edilir (50 test). `scripts/benchmark.py` gerçek bir tiny modelle, hem V1 hem V2 (karışık `status` dağılımlı) verilerle uçtan uca denenmiştir.

## API

> **Henüz uygulanmadı (Faz 6).**

`POST /extract` ve `GET /health` yalnızca yerel inference çalıştıktan sonra eklenecek.

---

## Test

```bash
pytest
```

Testler **gerçek bir LLM yüklemez**; chat template'i sahte bir tokenizer ile doğrulanır. Gerçek model inference'ı için opsiyonel bir entegrasyon testi Faz 4'te eklenecektir.

Lint / format:

```bash
ruff check .
ruff format .
```

---

## Mevcut Kısıtlar

- Gerçek veritabanı bağlantısı, SQL üretimi ve SQL çalıştırma **yoktur** — bilinçli olarak kapsam dışında.
- `target` ve `field` isimleri sentetik ve geçicidir; işverenin gerçek şeması geldiğinde değişecektir.
- Tarih normalizasyonu şu an dataset/model tarafındadır; ileride deterministik koda taşınabilir.
- Entity resolution (ör. `"Coca Cola"` → `company_id=187`) sonraki bir aşamadadır.
- QLoRA Apple Silicon'da kullanılamaz (`bitsandbytes` desteklemiyor).
- Deterministik generation ayarları rastgeleliği azaltır ama farklı donanım/backend'lerde matematiksel olarak birebir aynı davranışı garanti etmez. Sistem sözleşmesi bu nedenle yalnızca `temperature=0`'a değil, **doğrulamaya ve kısıtlı çıktı tasarımına** dayanır.
- Şema seviyesindeki çelişki kontrolü (`app/schemas/semantics.py`) yalnızca **aynı alan** içindeki çelişkileri yakalar (eq/eq, eq/neq, eq/in, sayısal/tarih aralık çelişkileri); farklı alanlar arası mantıksal tutarsızlıklar (ör. iş kuralı ihlalleri) kapsam dışıdır.
- V2 dataset'inin (`sql_json_dataset_v2_professional/`) `validation`/`test`/`challenge_test` dosyaları, üreticinin kendi kontrolünün kaçırdığı 54 satırlık büyük/küçük-harf-varyantı sızıntı için bağımsızca temizlenmiştir (bkz. Dataset bölümü); orijinaller `*.jsonl.orig` olarak durur.

---

## Güvenlik Notu

V1 SQL çalıştırmasa da tasarım ileriye dönük kurgulanmıştır: model değerleri **hiçbir zaman** doğrudan SQL string'ine birleştirilmeyecek. İleride parametreli sorgular, izin listeli alanlar/tablolar/operasyonlar ve deterministik bir SQL builder kullanılacak. LLM'in ürettiği tablo adlarına veya ham SQL'e asla güvenilmez.
