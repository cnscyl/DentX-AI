import cv2
import os
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from ultralytics import YOLO

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
def analyze_and_visualize(sender, instance, created, **kwargs):
    if created:
        try:
            image_path = instance.image.path
            
            # 1. CLAHE (Ön İşleme) - Resim zaten grileşti ve netleşti
            img = cv2.imread(image_path) # Çizim için renkli (BGR) okuyoruz
            
            # 2. YOLOv8 Analizi
            model = YOLO('yolov8n.pt') 
            results = model(image_path)

            found_something = False
            for result in results:
                if len(result.boxes) > 0:
                    found_something = True
                    # En yüksek güven skorunu al
                    instance.confidence_score = float(max(result.boxes.conf))
                    
                    # --- KUTU ÇİZME İŞLEMİ ---
                    for box in result.boxes:
                        # Koordinatları al
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Resmin üzerine kırmızı bir kutu çiz (BGR: 0,0,255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Üzerine güven skorunu yaz
                        label = f"Tespit: %{conf:.2f}"
                        cv2.putText(img, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Demo amaçlı bir bulguyu true yapalım
                    instance.has_caries = True 
            
            # 3. Üzerine çizim yapılmış resmi kaydet
            if found_something:
                cv2.imwrite(image_path, img)
                instance.save(update_fields=['confidence_score', 'has_caries'])
            
            print(f"Analiz ve Görselleştirme Tamamlandı.")

        except Exception as e:
            print(f"Hata oluştu: {e}")