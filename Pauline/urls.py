from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('backend.urls')),  # Ensure 'job.urls' is correctly spelled

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
