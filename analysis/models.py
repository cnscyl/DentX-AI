import cv2
import os
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from ultralytics import YOLO  # YOLO kütüphanesini ekledik

class PanoramicXRay(models.Model):
    image = models.ImageField(upload_to='xrays/') 
    upload_date = models.DateTimeField(auto_now_add=True)
    has_caries = models.BooleanField(default=False) 
    has_impaction = models.BooleanField(default=False) 
    has_bone_loss = models.BooleanField(default=False) 
    confidence_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"X-Ray {self.id} - {self.upload_date}"

@receiver(post_save, sender=PanoramicXRay)
def process_and_analyze_xray(sender, instance, created, **kwargs):
    if created:
        try:
            image_path = instance.image.path
            
            # --- 1. ADIM: CLAHE (ÖN İŞLEME) ---
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_img = clahe.apply(img)
                cv2.imwrite(image_path, enhanced_img)
                print(f"CLAHE Uygulandı.")

            # --- 2. ADIM: YOLOv8 ANALİZ (YAPAY ZEKA) ---
            # Şimdilik hazır eğitilmiş 'yolov8n.pt' (nano) modelini kullanıyoruz
            model = YOLO('yolov8n.pt') 
            results = model(image_path) # Resmi analiz et

            # Analiz sonuçlarını veritabanına yaz (Örnek mantık)
            for result in results:
                # Eğer model bir şeyler bulduysa (şimdilik demo amaçlı)
                if len(result.boxes) > 0:
                    instance.confidence_score = float(result.boxes.conf[0])
                    # Gerçek modelinde sınıflara göre has_caries vb. güncelleyeceğiz
                    instance.save()
            
            print(f"YOLOv8 Analizi Tamamlandı.")
            
        except Exception as e:
            print(f"Hata: {e}")