from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('alert_feed/', views.alert_feed, name='alert_feed'),
]
