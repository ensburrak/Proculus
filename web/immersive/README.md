# PRISMA / Proculus Immersive

Bu dizin, mevcut Python trading çekirdeğini değiştirmeden açılabilen, **ayrı ve izole bir web sunum katmanıdır**.

- `web/immersive/index.html`: Three.js ile gerçek geometri kullanan PRISMA manta/akış maskotu.
- Tam ekran kapatılabilir açılış, Türkçe isteğe bağlı TTS, imleçle etkileşim ve bağlamsal modül açıklamaları.
- İnternet bağlantısı olmadığında CSS 3D yedek görünümü.
- Bu dalda doğrulanmış bir Proculus web kontrol API sözleşmesi yoktur. Bu nedenle *gerçek emir, pozisyon, risk config düzenleme* gibi eylemler bilinçli olarak uygulanmamıştır.

## Lokal sunum

Depo kökünde:

```cmd
python -m http.server 8080 --directory web/immersive
```

Ardından tarayıcıda **localhost:8080** aç.

## Genişletilmiş HTML ile farkı

Ek olarak teslim edilen `Proculus_Immersive_v2.html`, daha büyük bir bağımsız çalışma stüdyosu ve çok sayıda simülasyon/kontrol içerir. Bu repo dizinindeki uygulama, mevcut depoya **güvenli, değişiklikten izole başlangıç entegrasyonudur**. İstenirse indirilen tam HTML burada `full.html` adıyla barındırılabilir. Tam HTML'nin simülasyon kontrolleri gerçek işlem motoruna bağlı değildir.

## Gerçek operasyonun sonraki aşaması

1. Backend servisinin bu dalda hangi giriş noktasından çalıştığını ve auth sözleşmesini doğrula.
2. API yüzeylerini OpenAPI ile tanımla, yetkili reverse proxy ve CORS ayarla.
3. Önce *okuma amaçlı* snapshot, telemetri ve denetim izi bağla.
4. Canlı mutasyonları ancak ayrı onay, kayıt ve risk testleri sonrasında sunucu tarafında uygula.

Statik HTML API anahtarını taşımaz; bu prototipten işlem emri gönderilmez.
