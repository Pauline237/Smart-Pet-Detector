from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings




urlpatterns = [
    
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload'),
    path('save-result/', views.save_result, name='save_result'),
   # path('about/', views.about, name='about'),
   # path('history/', views.history, name='history'),  



]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)