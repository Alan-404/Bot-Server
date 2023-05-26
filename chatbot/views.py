from django.http import JsonResponse
from rest_framework.decorators import api_view

from .services import ChatbotService

chatbot_service = ChatbotService()

# Create your views here.
@api_view(['GET', "POST"])
def get_message(request):
    if request.method == "POST":
        return JsonResponse({"response": chatbot_service.response(request.data['message'])})