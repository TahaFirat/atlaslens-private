# AtlasLens yatırımcı özel demo runbook'u

Bu akış, AtlasLens'in mevcut ve gerçek yerel kabiliyetlerini 3–5 dakikalık özel
bir ürün anlatımında birleştirir. Ankara Mapillary pilot referans indeksi ve
MegaLoc yerel olarak çalışır; vaka, kanıt, hipotez, operatör değerlendirmesi ve
audit kayıtları mevcut kanonik servislerde tutulur. NVIDIA ve başka bir bulut
sağlayıcısı demo için gerekli değildir. Normal başlatma, yalnızca kullanıcının
görüntülediği viewport için anahtarsız OpenStreetMap raster tile'larını yükler;
ağ yoksa adaylar ve belirsizlik geometrisi yerel fallback üzerinde kalır.

Bu bir doğruluk benchmark'ı, Türkiye-geneli ürün iddiası, kalibre edilmiş
olasılık, hukuken sertifikalı delil veya üretim/hukuk onayı değildir. Pilot
referans kapsamı Ankara ile sınırlıdır. Mapillary materyali AtlasLens'e ait
değildir; kaynak ve CC-BY-SA-4.0 attribution bilgileri korunur.

## Kesin başlatma ve kapatma komutları

Repository kökünden:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-investor-demo.ps1
```

Başka bir dizinden:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\geoSearch\scripts\start-investor-demo.ps1
```

Başarı yalnızca disk, port, yerel runtime, MegaLoc gerçek-inference readiness,
pilot bundle bütünlüğü, kanonik vaka hazırlığı, API health/readiness, özel indeks
durumu, web HTTP 200 ve API proxy kontrollerinden sonra yazılır. Başlatıcı,
hazırladığı kanonik vakanın doğrulanmış UUID'sini içeren kesin URL'yi `Demo:`
satırında yazar. Yalnızca o satırdaki URL'yi açın; biçimi:

```text
http://127.0.0.1:5173/?demo=investor&lang=tr&caseId=<generated-UUID>
```

`caseId` olmayan eski bağlantı çökmez; uyarılı açılış ekranında mevcut vaka
API'sinden gelen yetkili vakalardan birinin açıkça seçilmesini ister. Hatalı veya
yinelenen `caseId` API'ye gönderilmez. Biçimsel olarak geçerli fakat bulunamayan
bir UUID açık bir “İstenen vaka bulunamadı” görünümü, **Yeniden dene** ve geri
dönüş eylemleri sunar.

Kapatma:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-investor-demo.ps1
```

Kapatıcı yalnızca doğrulanmış PID, başlangıç zamanı, komut, parent/child ve port
sahipliği kayıtları bu demo ile eşleşen süreçleri durdurur. Ardından tam olarak
`.local\run\investor-demo` altındaki özel DB, geçici upload, log ve state
alanını containment kontrolüyle siler. Bu runtime verisi geri alınamaz; harici
`C:\AtlasLensPilot\mapillary-demo` pilot materyaline dokunulmaz.

## Demo öncesi kontrol listesi

- Demo makinesi özel ve güvenilir bir operatör oturumunda olmalı.
- C: sürücüsünde en az 35 GiB, D: sürücüsünde en az 8 GiB boş alan olmalı.
- `C:\AtlasLensPilot\mapillary-demo` altındaki retained varlıklar, attribution
  sidecar'ları ve yayınlanmış indeks değişmemiş olmalı.
- Kurulu MegaLoc runtime'ı ve frontend `node_modules` mevcut olmalı. Başlatıcı
  eksikleri indirmez veya kurmaz.
- 8794, 8000 ve 5173 portlarında başka listener olmamalı. Başlatıcı çakışan
  süreci kapatmaz; fail-closed durur.
- Daha önce yarım kalmış bir çalıştırma varsa önce kapatma komutu çalıştırılmalı.
- Ekran paylaşımında terminal logları, yerel dosya gezgini veya pilot varlık
  dizini gösterilmemeli.
- Tarayıcıda yalnızca başlatıcının yazdığı loopback URL açılmalı.
- İnternet ve NVIDIA kapalı kalabilir. `NVIDIA_API_KEY`, `OPENAI_API_KEY` ve
  `MAPILLARY_ACCESS_TOKEN` gerekmez; demo çocuk süreçlerinde boşaltılır.
- Online alt harita API anahtarı gerektirmez. Başlatıcı tek merkezî
  `VITE_MAP_TILE_URL` değeri olarak
  `https://tile.openstreetmap.org/{z}/{x}/{y}.png` kullanır; bulk indirme,
  prefetch veya offline tile arşivi oluşturmaz. Attribution her durumda görünür.

## Beklenen ekranlar ve tıklama sırası

1. Başlatıcının yazdığı `caseId` içeren URL hazır vakayı doğrudan harita-merkezli
   çalışma alanında açar. Üst çubukta analiz durumunu, “Ankara referans pilotu”
   kapsamını ve yalnızca yerel çalışma durumunu gösterin.
2. Eski, `caseId` içermeyen bağlantı kullanılmışsa otomatik veya başlığa dayalı
   vaka seçimi yapılmaz; **Diğer yetkili vakalar** listesini açıp doğru vakayı
   açıkça seçin. Kısa anlatım için başlatıcının kesin bağlantısı tercih edilir.
3. Sol panelde silinmiş-görsel privacy placeholder'ını, kaynağı, metadata'nın
   çıkarıma verilmediğini ve mevcut retention durumunu gösterin.
4. Haritada numaralı adayları, top-1 ayrımını, belirsizlik katmanını, zoom/scale,
   **Adaylara sığdır** ve katman kontrolünü gösterin. Kart ve marker seçiminin
   iki yönde senkron olduğunu doğrulayın.
5. Sağ panelde bunun yalnızca “Ankara referans koleksiyonu içinde görsel benzerlik
   araması” olduğunu gösterin. Ham cosine similarity confidence/olasılık değildir;
   hazır vaka UI'a gömülü değildir ve başlatmada gerçek yerel sorgudan üretilir.
6. Kapsam veya provider yetersizse aday yerine açık çekimserlik görünür. Generic
   upload yalnız Ankara indeksi nedeniyle Ankara sonucu üretemez.
7. Alt **Kanıt ve teknik ayrıntılar** drawer'ını açıp attribution/provenance,
   operatör değerlendirmesi ve audit bütünlüğünü gösterin. Sırf ekranı doldurmak
   için kabul/ret üretmeyin.

## 3–5 dakikalık Türkçe konuşma metni

**00:00–00:30 — Çerçeve**

“AtlasLens bugün tek komutla, tamamen bu makinede açıldı. İnternete veya NVIDIA
başarısına bağlı değil. Göreceğiniz şey Türkiye-geneli doğruluk iddiası değil;
Ankara'daki küçük ve attribution'ı korunmuş özel pilotu, mevcut inceleme ürününe
bağlayan dürüst bir teknik demo.”

**00:30–01:05 — Yetki ve amaç**

“Her inceleme bir sonuçla değil, kaynak, yetki ve amaç kaydıyla başlıyor. Bu hazır
vaka yalnızca izinli Mapillary pilot materyalini kullanıyor. Görselin kendisi vaka
DB'sinde tutulmuyor; yan dosyadaki gerçek koordinat modele verilmiyor. Kaynak
bağlamı, saklama politikası ve yetki beyanı audit geçmişine giriyor.”

**01:05–01:45 — Yerel analiz ve sağlayıcılar**

“Başlatma sırasında görsel byte'ları yalnızca loopback üzerindeki yerel MegaLoc
worker'ına gitti ve checksum ile doğrulanan Ankara pilot indeksinde arandı. Burada
analiz modunun yalnızca yerel olduğunu, kullanılan kaynakları ve provider
durumlarını görüyoruz. NVIDIA opsiyonel ve deneysel; kapalı veya hatalı olduğunda
yerel inceleme devam ediyor.”

**01:45–02:40 — Harita, belirsizlik ve kanıt**

“Sistem bir eşleşme bulduysa bunu kesin konum diye değil, haritada pozitif ve
muhafazakâr belirsizlik yarıçapına sahip hipotezler olarak gösteriyor. Sıralama
ham retrieval benzerliğinden geliyor; bu değer kalibre edilmiş olasılık değil.
Her adayın attribution'ı, provenance'ı ve dayandığı kanıt kaydı var. Yeterli sinyal
yoksa sistemin doğru cevabı çekimser kalmak.”

**02:40–03:30 — Operatör değerlendirmesi**

“Model sonucu kararın kendisi değil. Operatör hipotezi inceleyip gerekçeli kabul,
ret veya belirsiz kararı ekleyebilir. Karar append-only geçmişe girer; sonradan
değişiklik yapılırsa önceki kayıt silinmez, yeni kayıt onu açıkça supersede eder.”

**03:30–04:15 — Audit ve dürüst sınırlar**

“Burada vaka olaylarının hash-zinciri bütünlüğünü doğruluyoruz. Bu, uygulama
seviyesinde kurcalamayı görünür kılan bir kontroldür; hukuken sertifikalı delil
değildir. Aynı şekilde Ankara pilotu Türkiye kapsamını, confidence değerleri de
kalibre edilmiş olasılığı kanıtlamaz. Ürün bugün incelemeyi izlenebilir ve dürüst
hale getiriyor; henüz geniş ölçekli doğruluk veya üretim servisi iddia etmiyor.”

**04:15–04:40 — Yatırım tezi**

“Yatırımın en yüksek kaldıraç alanları; hakları net daha geniş ve temsilî bir
referans/evaluation korpusu, bağımsız kalibrasyon ve adalet çalışması, üretim
kimliği ve yetkilendirme, worker supervision ve ölçülebilir SLO'lar. Bugünkü demo
bu yatırımların üzerine kurulacağı denetlenebilir ürün omurgasını gösteriyor.”

## İnternet ve NVIDIA kapalı fallback akışı

1. İnterneti ve NVIDIA'yı açmaya çalışmayın; normal demo komutunu çalıştırın.
2. Başlatıcı pilot bundle ve yerel MegaLoc hazırsa aynı kanonik vakayı üretir.
3. Tile yüklemesi başarısızsa çalışma alanı **Alt harita çevrimdışı** durumunu ve
   **Alt haritayı yeniden dene** kontrolünü gösterir. Hafif yerel fallback,
   numaralı adaylar, belirsizlik katmanı ve daima görünür OpenStreetMap
   attribution kalır; fallback ayrıntılı bir yol haritası değildir.
4. Metinsel aday sırası, kapsam/benzerlik semantiği, evidence ve audit kayıtları
   asıl inceleme görünümüdür; arka plan tile'ı aday kanıtını değiştirmez.
5. Yerel MegaLoc veya pilot bundle readiness'i başarısızsa sonucu taklit etmeyin.
   Başlatıcının fail-closed hatasını gösterin ve önceden alınmış gerçek sonuç diye
   ekran görüntüsü sunmayın. Runbook'taki readiness sorununu giderip yeniden
   çalıştırın.

## Bilinen sınırlamalar ve yatırımın çözebileceği darboğazlar

- Pilot yalnızca 29 Ankara referansı ve 11 kilitli holdout sorgusundan oluşur;
  coğrafi kapsama ve temsil gücü düşüktür.
- Ham cosine benzerliği ve göreli sıralama kalibre edilmiş confidence değildir;
  kanonik görünüm model hipotezleri için muhafazakâr en az 25 km yarıçap kullanır.
- Pilotun önceki küçük benchmark gözlemleri ürün doğruluğu, Türkiye kapsamı veya
  ticari uygunluk iddiasına dönüştürülemez.
- Mapillary hak/attribution kaydı özel teknik demoya kabul sağlar; profesyonel
  hukuk incelemesi ve kamusal/üretim dağıtım izni yerine geçmez.
- Worker, API ve web tek Windows hostunda geliştirici süreçleridir; üretim
  supervisor'ı, kimlik/yetkilendirme, dağıtık queue, SLO veya HA değildir.
- Audit hash zinciri uygulama bütünlüğünü kontrol eder; güvenilir zaman damgası,
  bağımsız imza, delil saklama kurumu veya hukuki sertifikasyon değildir.
- Temsilî, lisanslı ulusal korpus; bağımsız train/validation/test ayrımı;
  kalibrasyon, bölgesel/adversarial/fairness değerlendirmesi ve operasyonel
  güvenlik yatırımı hâlâ gereklidir.

## Sorun giderme

- **Disk gate:** C: veya D: eşiğini karşılamadan devam etmeyin; model/dataset
  silme veya indirme işlemini bu runbook kapsamında yapmayın.
- **Port conflict:** Hata mesajındaki sürecin gerçekten size ait olduğunu ayrıca
  inceleyin. Başlatıcı onu kapatmaz. Demo portlarını serbest bırakmadan tekrar
  çalıştırmayın.
- **Pilot/index gate:** Dosya, hash, attribution veya rights sidecar eksikse
  indirme yapmayın. Harici pilot kaynağını yetkili bakım süreciyle geri yükleyin.
- **Worker readiness:** Mevcut yerel MegaLoc runtime'ını onarın; başka model
  indirmeyin veya sentetik sonucu gerçek gibi göstermeyin.
- **Yarım startup:** Kapatma komutunu çalıştırın. Güvenilir süreç kaydı yoksa
  kapatıcı bilinmeyen listener'lara dokunmaz. Kapatıcı, launcher/listener
  başlangıç kimliklerini süreç ağacını tararken ve ilk durdurmadan hemen önce
  yeniden doğrular; Windows'un eski parent-PID kayıtları sahiplik sayılmaz.
- **Vaka bağlantısı:** Terminaldeki `Demo:` satırını yeniden kopyalayın. Eksik veya
  hatalı `caseId` için uyarılı yetkili-vaka seçimini kullanın; geçerli ama bulunmayan
  UUID'de **Yeniden dene** yalnızca aynı kesin vaka isteğini tekrarlar.
