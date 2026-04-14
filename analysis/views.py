from django.shortcuts import render, redirect
from .models import PanoramicXRay
from .forms import XRayUploadForm

def dashboard(request):
    if request.method == 'POST':
        form = XRayUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = XRayUploadForm()
    
    xrays = PanoramicXRay.objects.all().order_by('-upload_date')
    return render(request, 'analysis/dashboard.html', {'xrays': xrays, 'form': form})