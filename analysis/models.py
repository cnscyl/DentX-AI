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

# --- AYARLAR ---
NAMES = ['Caries', 'Filling', 'Periapical Lesion', 'Root Canal Treatment', 'Impacted Tooth']

# Sınıf Renkleri (BGR)
COLOR_PALETTE = {
    'Caries': (0, 0, 255),               # Kırmızı
    'Filling': (255, 0, 0),              # Mavi
    'Periapical Lesion': (0, 165, 255),  # Turuncu
    'Root Canal Treatment': (255, 0, 255), # Mor
    'Impacted Tooth': (0, 255, 255)      # Sarı
}

@receiver(post_save, sender=PanoramicXRay)
def analyze_and_visualize(sender, instance, created, **kwargs):
    if created:
        try:
            image_path = instance.image.path
            
            # Görüntüyü Oku
            img = cv2.imread(image_path)
            if img is None: return

            # Modeli Yükle
            model = YOLO('bestv4.pt') 
            
            # --- KRİTİK GÜNCELLEME: conf=0.50 ---
            # Model sadece %50 ve üzeri emin olduğu tespitleri döndürecek.
            # Hatalı "Impacted" etiketlerini bu temizler.
            results = model(image_path, conf=0.50) 

            found_something = False
            detected_caries = False
            detected_impaction = False
            max_conf = 0.0

            for result in results:
                for box in result.boxes:
                    found_something = True
                    class_id = int(box.cls[0])
                    
                    if class_id < len(NAMES):
                        label_name = NAMES[class_id]
                    else:
                        label_name = "Tespit"

                    conf = float(box.conf[0])
                    if conf > max_conf: max_conf = conf

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # --- GÖRSELLEŞTİRME ---
                    color = COLOR_PALETTE.get(label_name, (0, 255, 0))

                    # Çerçeve
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Yazı (Arka plansız, temiz font)
                    label_text = f"{label_name} %{conf:.2f}"
                    cv2.putText(img, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    # Boolean Mantığı
                    name_lower = label_name.lower()
                    if 'caries' in name_lower: detected_caries = True
                    if 'impacted' in name_lower: detected_impaction = True

            if found_something:
                cv2.imwrite(image_path, img)
                instance.confidence_score = round(max_conf, 2)
                instance.has_caries = detected_caries
                instance.has_impaction = detected_impaction
                instance.save(update_fields=['confidence_score', 'has_caries', 'has_impaction'])
            
            print(f"Analiz Tamamlandı ID: {instance.id}")

        except Exception as e:
            print(f"Hata: {e}")