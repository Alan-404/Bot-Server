from django.http import JsonResponse
from rest_framework.decorators import api_view
from .service import STTService
import librosa
import io
# Create your views here.

stt_service = STTService()

@api_view(["POST"])
def file(request):
    samples, _ = librosa.load(io.BytesIO(request.data['file'].read()), sr=16000)
    return JsonResponse({"result": stt_service.transform_samples(samples)})

@api_view(['POST'])
def voice(request):
    return JsonResponse({"result": stt_service.transform_samples(request.data['samples'])})