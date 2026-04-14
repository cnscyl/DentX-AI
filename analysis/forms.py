from django import forms
from .models import PanoramicXRay

class XRayUploadForm(forms.ModelForm):
    class Meta:
        model = PanoramicXRay
        fields = ['image'] # Sadece resim seçme alanı