from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings




urlpatterns = [
  #  path('', views.home, name='homepage'),
  

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)