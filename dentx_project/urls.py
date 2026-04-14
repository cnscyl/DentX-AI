from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from analysis.views import dashboard

urlpatterns = [
    path('admin/', admin.site.admin_view(admin.site.urls)),
    path('dashboard/', dashboard, name='dashboard'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)