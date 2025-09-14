from django.urls import path
from .views import RecommendInternships, HomePageView, InternshipAnalyticsView, PortalPageView

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('portal/', PortalPageView.as_view(), name='portal'),
    path('api/recommend/', RecommendInternships.as_view(), name='recommend-internships'),
    path('api/analytics/', InternshipAnalyticsView.as_view(), name='internship-analytics'),
]