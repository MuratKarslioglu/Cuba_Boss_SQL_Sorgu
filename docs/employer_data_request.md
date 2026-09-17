# İşverenden Veri Talebi — Kontrol Listesi

Bu doküman, Türkçe → SQL-JSON çıkarım sisteminin sentetik/prototip aşamadan gerçek üretim verisine geçmesi için işverenden talep edilmesi gereken bilgi ve verileri listeler. Şu ana kadar model **hiç gerçek kullanıcı sorusu ve hiç gerçek veritabanı şeması görmedi** — mevcut eğitim verisi (`datasets/sql_json_dataset_v2_professional/`) tamamen sentetik olarak üretildi (bkz. `docs/text_to_sql_json_project_spec.md` §35). Aşağıdaki maddeler ne kadar eksiksiz doldurulursa, sistemin gerçek kullanımda ne kadar güvenilir olacağı o kadar netleşir.

Her bölüm bağımsız doldurulabilir; eksik bırakılan bölümler proje ilerlerken tekrar sorulacaktır.

---

## 1. Amaç

Şu an elimizdeki prototip, jenerik "satış" alanında (şirket/ürün/müşteri/tarih vb.) sentetik olarak üretilmiş ~18.000 örnekle eğitildi ve test edildi. Bu veri gerçekçi görünse de gerçek kullanıcıların nasıl soru sorduğunu, gerçek veritabanınızın şemasını ve gerçek iş terminolojinizi yansıtmıyor. Aşağıdaki bilgiler, sistemi sizin gerçek ortamınıza uyarlamak için gerekli.

---

## 2. Veritabanı Şeması

- [ ] Tablo adları, kolon adları ve tipleri (bir `CREATE TABLE` dökümü veya ER diyagramı yeterli)
- [ ] Primary key / foreign key ilişkileri
- [ ] Raporlama için zaten kullanılan view'lar varsa listesi
- [ ] **Kritik soru:** Sorgular tek bir tablo üzerinden mi yanıtlanabiliyor, yoksa join gerektiren normalize edilmiş çoklu tablo yapısı mı var? (Bu, mevcut tek-hedefli JSON sözleşmesinin yeterli olup olmadığını doğrudan belirler.)

## 3. Gerçek Örnek Promptlar / Loglar

- [ ] En az **200-300**, tercihen **500+** gerçek kullanıcı sorusu (destek talepleri, analistlere Slack/e-posta ile gelen istekler, arama logları, veya iş kullanıcılarına dikte ettirilmiş örnekler)
- [ ] **Ham, düzeltilmemiş haliyle** istenir — yazım hatası, argo, eksik cümle dahil. Sentetik veride tam olarak bu eksik.
- [ ] SQL veya cevap **gerekmiyor**, yalnızca doğal dil soru metni yeterli (bu, işverenin işini kolaylaştırır)

## 4. Teslim Formatı

- [ ] CSV/Excel (her satırda bir soru) veya JSONL kabul edilir
- [ ] Kişisel/hassas veri içeriyorsa bölüm 9'daki anonimleştirme kısıtlarına bakınız

## 5. İş Terminolojisi Sözlüğü

- [ ] İş terimi → veritabanı kolonu eşlemesi (örnek tablo aşağıda)
- [ ] Satış ekibinin/kullanıcıların gerçekte kullandığı eş anlamlılar/kısaltmalar

| İş terimi | Eş anlamlılar | Veritabanı kolonu |
|---|---|---|
| Ciro | toplam satış tutarı, hasılat | `?` |
| ... | ... | `?` |

## 6. Entity-ID Eşlemesi

- [ ] Her filtrelenebilir alan (şirket, müşteri, ürün, vb.) için: gerçek tanımlayıcı **isim** mi, **kod** mu, yoksa iç **numeric ID** mi kullanılıyor?
- [ ] İsimden ID'ye çözümleme için bir lookup tablosu/servis var mı? (İleride "Coca Cola" → `company_id=187` gibi bir Entity Resolver aşaması bu bilgiye dayanacak, bkz. spec §35)

## 7. İzin Verilen Filtre / Sorgu Türleri

Aşağıdakiler şu an desteklenen sorgu tiplerinin sade dile çevrilmiş hali — hangileri gerçekten ihtiyaç, hangileri gereksiz?

- [ ] Toplam/ortalama/sayım istenebilmeli mi? (sum/average/count)
- [ ] En yüksek/en düşük N kayıt (top-N/bottom-N) istenebilmeli mi?
- [ ] Tarih aralığı sorguları gerekli mi? Hangi sıklıkta?
- [ ] Gruplama (örn. "ürüne göre") gerekli mi?
- [ ] Çelişen koşullar girildiğinde (örn. "X firması VE Y firması" aynı alanda) sistem ne yapmalı — reddetmeli mi, açıklama mı istemeli?

## 8. Tarih Semantiği

- [ ] "Bu ay" ifadesi neye göre hesaplanmalı — takvim ayı mı, mali yıl mı?
- [ ] Hangi saat dilimi kullanılıyor?
- [ ] Geçmiş veri hangi tarihten itibaren mevcut?
- [ ] "Son 3 ay", "geçen hafta" gibi göreli ifadeler, varsa mevcut bir sistemde zaten nasıl yorumlanıyor?

## 9. Gizlilik / Anonimleştirme Kısıtları

- [ ] Gerçek müşteri adı/ID'lerinin bize teslim edilmeden önce maskelenmesi gerekiyor mu?
- [ ] NDA/DPA (veri işleme sözleşmesi) gerekli mi?
- [ ] Veri nerede saklanabilir/saklanamaz (yalnızca işveren altyapısı, bulut kısıtı vb.)?

## 10. Minimum Hacim / Kabul Kriteri

| Kalem | Minimum | Tercih Edilen |
|---|---|---|
| Veritabanı şeması | Eksiksiz tablo/kolon listesi | + ER diyagramı |
| Gerçek örnek promptlar | 200 | 500+ |
| Terminoloji sözlüğü | Tüm filtrelenebilir kolonları kapsar | + eş anlamlılar |
| Entity-ID eşlemesi | En az ana filtre alanları için (şirket/ürün) | Tüm alanlar için |

---

*Bu doküman `docs/text_to_sql_json_project_spec.md` §35 "Future Production Extension" bölümüne ve `app/schemas/vocabulary.py`'deki geçici sentetik alan/hedef notuna dayanılarak hazırlanmıştır.*
