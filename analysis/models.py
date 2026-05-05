import cv2
import os
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from ultralytics import YOLO
from django.conf import settings

class PanoramicXRay(models.Model):
    # Orijinal resim (Değiştirilmeden saklanacak)
    image = models.ImageField(upload_to='xrays/originals/') 
    # Analiz edilmiş resim (Kutulu hali buraya kaydedilecek)
    processed_image = models.ImageField(upload_to='xrays/processed/', null=True, blank=True)
    
    upload_date = models.DateTimeField(auto_now_add=True)
    has_caries = models.BooleanField(default=False) 
    has_impaction = models.BooleanField(default=False) 
    has_crown = models.BooleanField(default=False) # Yeni eklendi
    has_bone_loss = models.BooleanField(default=False) 
    confidence_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"X-Ray {self.id} - {self.upload_date}"

# --- AYARLAR ---
NAMES = [
    'Bone Loss', 'Caries', 'Crown', 'Cyst', 'Filling', 'Fracture teeth', 
    'Implant', 'Malaligned', 'Mandibular Canal', 'Missing teeth', 
    'Periapical lesion', 'Permanent Teeth', 'Primary teeth', 'Retained root', 
    'Root Canal Treatment', 'Root Piece', 'Root resorption', 'Supra Eruption', 
    'TAD', 'abutment', 'attrition', 'bone defect', 'gingival former', 
    'impacted tooth', 'maxillary sinus', 'metal band', 'orthodontic brackets', 
    'permanent retainer', 'plating', 'post - core', 'wire'
]

# Crown listeye eklendi, böylece çizim yapılacak
TARGET_CLASSES = ['Caries', 'Filling', 'Periapical lesion', 'Root Canal Treatment', 'impacted tooth', 'Crown']

COLOR_PALETTE = {
    'Caries': (0, 0, 255),               # Kırmızı
    'Filling': (255, 0, 0),              # Mavi
    'Periapical lesion': (0, 165, 255),  # Turuncu
    'Root Canal Treatment': (255, 0, 255),# Mor
    'impacted tooth': (0, 255, 255),     # Sarı
    'Crown': (0, 255, 0)                 # Yeşil (Yeni eklendi)
}

@receiver(post_save, sender=PanoramicXRay)
def analyze_and_visualize(sender, instance, created, **kwargs):
    if created:
        try:
            # 1. Dosya Yollarını Hazırla
            original_path = instance.image.path
            img = cv2.imread(original_path)
            if img is None: return

            # Analiz için görüntünün bir kopyasını al
            processed_img = img.copy()

            # 2. Modeli Yükle ve Tahmin Et
            model = YOLO('bestv6.pt') 
            results = model.predict(
                source=original_path, 
                conf=0.10,           
                iou=0.25,            
                imgsz=1024,           
                augment=False,       
                agnostic_nms=True    
            )

            found_something = False
            detected_caries = False
            detected_impaction = False
            detected_crown = False
            max_conf = 0.0

            # 3. Sonuçları İşle
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    if class_id < len(NAMES):
                        label_name = NAMES[class_id]
                    else:
                        continue

                    if label_name in TARGET_CLASSES:
                        found_something = True
                        conf = float(box.conf[0])
                        if conf > max_conf: max_conf = conf

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = COLOR_PALETTE.get(label_name, (0, 255, 0))

                        # Çizimi KOPYA resim üzerine yap
                        cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 3)
                        label_text = f"{label_name} %{conf:.2f}"
                        cv2.putText(processed_img, label_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                        nl = label_name.lower()
                        if 'caries' in nl: detected_caries = True
                        if 'impacted' in nl: detected_impaction = True
                        if 'crown' in nl: detected_crown = True

            # 4. İşlenmiş Resmi Kaydet ve DB Güncelle
            if found_something:
                filename = os.path.basename(original_path)
                processed_filename = f"analysed_{filename}"
                relative_path = os.path.join('xrays/processed/', processed_filename)
                full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                cv2.imwrite(full_path, processed_img)

                instance.processed_image = relative_path
                instance.confidence_score = round(max_conf, 2)
                instance.has_caries = detected_caries
                instance.has_impaction = detected_impaction
                instance.has_crown = detected_crown
                instance.save(update_fields=['processed_image', 'confidence_score', 'has_caries', 'has_impaction', 'has_crown'])
            
            print(f"Analiz Tamamlandı - ID: {instance.id}")

        except Exception as e:
            print(f"Hata Oluştu: {e}")