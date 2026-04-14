from django.contrib import admin
from .models import PanoramicXRay # Oluşturduğumuz tabloyu çağırıyoruz

@admin.register(PanoramicXRay)
class PanoramicXRayAdmin(admin.ModelAdmin):
    # Liste sayfasında hangi bilgilerin görüneceğini seçiyoruz
    list_display = ('id', 'upload_date', 'has_caries', 'has_bone_loss', 'has_impaction')