from django.conf.urls import url


from . import views

urlpatterns = [
    url(r'^form/$',views.request_page ,name='home'),
    url(r'^force/$',views.formulate ,name='formulate'),
]
