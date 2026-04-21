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

# --- AYARLAR: data.yaml İLE BİREBİR AYNI SIRALAMA (ID KAYMASINI ÖNLER) ---
NAMES = [
    'Bone Loss', 'Caries', 'Crown', 'Cyst', 'Filling', 'Fracture teeth', 
    'Implant', 'Malaligned', 'Mandibular Canal', 'Missing teeth', 
    'Periapical lesion', 'Permanent Teeth', 'Primary teeth', 'Retained root', 
    'Root Canal Treatment', 'Root Piece', 'Root resorption', 'Supra Eruption', 
    'TAD', 'abutment', 'attrition', 'bone defect', 'gingival former', 
    'impacted tooth', 'maxillary sinus', 'metal band', 'orthodontic brackets', 
    'permanent retainer', 'plating', 'post - core', 'wire'
]

# Dashboard'da işaretlenmesini istediğin 5 ana sınıf (Küçük harf duyarlı)
TARGET_CLASSES = ['Caries', 'Filling', 'Periapical lesion', 'Root Canal Treatment', 'impacted tooth']

# Sınıf Renkleri (BGR Formatı)
COLOR_PALETTE = {
    'Caries': (0, 0, 255),               # Kırmızı
    'Filling': (255, 0, 0),              # Mavi
    'Periapical lesion': (0, 165, 255),  # Turuncu
    'Root Canal Treatment': (255, 0, 255), # Mor
    'impacted tooth': (0, 255, 255)      # Sarı
}

@receiver(post_save, sender=PanoramicXRay)
def analyze_and_visualize(sender, instance, created, **kwargs):
    if created:
        try:
            image_path = instance.image.path
            img = cv2.imread(image_path)
            if img is None: return

            # En son eğittiğin modeli yükle
            model = YOLO('bestv4.pt') 
            
            # conf=0.50 ile hatalı düşük tahminleri eliyoruz
            results = model(image_path, conf=0.25)

            found_something = False
            detected_caries = False
            detected_impaction = False
            max_conf = 0.0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    # Güvenlik Kontrolü: ID 31'den küçükse ismi al
                    if class_id < len(NAMES):
                        label_name = NAMES[class_id]
                    else:
                        continue # Tanımsız ID gelirse atla

                    # FİLTRE: Sadece seçtiğimiz 5 sınıfı işle
                    if label_name in TARGET_CLASSES:
                        found_something = True
                        conf = float(box.conf[0])
                        if conf > max_conf: max_conf = conf

                        # Koordinatları Al
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Renk Belirle
                        color = COLOR_PALETTE.get(label_name, (0, 255, 0))

                        # Çizim (Kutu ve Yazı)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        label_text = f"{label_name} %{conf:.2f}"
                        cv2.putText(img, label_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                        # Veritabanı Boolean Mantığı
                        nl = label_name.lower()
                        if 'caries' in nl: detected_caries = True
                        if 'impacted' in nl: detected_impaction = True

            # Kayıt İşlemi
            if found_something:
                cv2.imwrite(image_path, img)
                instance.confidence_score = round(max_conf, 2)
                instance.has_caries = detected_caries
                instance.has_impaction = detected_impaction
                instance.save(update_fields=['confidence_score', 'has_caries', 'has_impaction'])
            
            print(f"Başarılı Analiz - ID: {instance.id}")

        except Exception as e:
            print(f"Hata Oluştu: {e}")