import cv2
import os
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from ultralytics import YOLO # Yapay zeka kütüphanesi

class PanoramicXRay(models.Model):
    image = models.ImageField(upload_to='xrays/') 
    upload_date = models.DateTimeField(auto_now_add=True)
    
    # Raporundaki bulgular
    has_caries = models.BooleanField(default=False) 
    has_impaction = models.BooleanField(default=False) 
    has_bone_loss = models.BooleanField(default=False) 
    confidence_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"X-Ray {self.id} - {self.upload_date}"

@receiver(post_save, sender=PanoramicXRay)
def analyze_xray(sender, instance, created, **kwargs):
    if created:
        try:
            image_path = instance.image.path
            
            # --- 1. ADIM: CLAHE (Ön İşleme) ---
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_img = clahe.apply(img)
                cv2.imwrite(image_path, enhanced_img)
            
            # --- 2. ADIM: YOLOv8 ANALİZ (Multitask Detection) ---
            # Şimdilik hazır yolov8n.pt modelini kullanıyoruz. 
            # Kendi modelini eğittiğinde buraya 'best.pt' yazacaksın.
            model = YOLO('yolov8n.pt') 
            results = model(image_path)

            # Analiz sonuçlarını işle
            for result in results:
                if len(result.boxes) > 0:
                    # En yüksek güven skorunu alalım
                    max_conf = max(result.boxes.conf).item()
                    instance.confidence_score = round(max_conf, 2)
                    
                    # Örnek: Eğer model 'person' (veya senin modelinde 'caries') bulursa
                    # Şimdilik demo için bir tanesini işaretleyelim
                    instance.has_caries = True 
                    
                    # Değişiklikleri kaydet (save_base döngüye girmemesi için önemli)
                    instance.save(update_fields=['confidence_score', 'has_caries'])
            
            print(f"Yapay Zeka Analizi Tamamlandı: {image_path}")

        except Exception as e:
            print(f"Analiz hatası: {e}")