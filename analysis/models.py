import cv2  # OpenCV kütüphanesi
import os   # Dosya yolları için gerekli modül
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

class PanoramicXRay(models.Model):
    # Röntgenin yüklendiği dosya yolu [cite: 541]
    image = models.ImageField(upload_to='xrays/') 
    # Yükleme tarihi [cite: 541]
    upload_date = models.DateTimeField(auto_now_add=True)
    # Raporunda belirtilen analiz bulguları [cite: 183, 409, 543]
    has_caries = models.BooleanField(default=False) 
    has_impaction = models.BooleanField(default=False) 
    has_bone_loss = models.BooleanField(default=False) 
    # Yapay zeka güven skoru [cite: 425, 543]
    confidence_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"X-Ray {self.id} - {self.upload_date}"

# --- RAPORDA BELİRTİLEN CLAHE ÖN İŞLEME FONKSİYONU [cite: 158, 400, 566] ---

@receiver(post_save, sender=PanoramicXRay)
def process_xray_on_upload(sender, instance, created, **kwargs):
    """
    Röntgen veritabanına kaydedildikten hemen sonra çalışır.
    CLAHE yöntemini uygulayarak görüntüyü netleştirir[cite: 400, 566].
    """
    if created: # Sadece yeni kayıt oluşturulduğunda çalışsın
        try:
            image_path = instance.image.path
            
            # 1. Görüntüyü gri tonlamalı oku [cite: 400]
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # 2. CLAHE objesini oluştur (clipLimit: 2.0, tileGridSize: 8x8) [cite: 400, 566]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                
                # 3. Filtreyi uygula [cite: 400]
                enhanced_img = clahe.apply(img)
                
                # 4. Orijinal dosyanın üzerine netleşmiş halini kaydet
                cv2.imwrite(image_path, enhanced_img)
                print(f"CLAHE işlemi başarıyla uygulandı: {image_path}") [cite: 566]
        except Exception as e:
            print(f"Görüntü işleme sırasında hata oluştu: {e}")