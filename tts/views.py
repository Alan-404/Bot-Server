from django.http import JsonResponse, FileResponse, HttpResponse
from rest_framework.decorators import api_view
import torch
from .services import TTSService
import wave
import io
import numpy as np
from scipy.io import wavfile
tts_service = TTSService()
# Create your views here.
@api_view(['POST'])
def transform(request):
    audio_data = tts_service.transform(request.data['text'])
    rate = 22050
    byte_stream = io.BytesIO()

    wavfile.write(byte_stream, rate, audio_data)
    torch.cuda.empty_cache()
    response = HttpResponse(byte_stream, content_type="audio/wav")

    response['Content-Disposition'] = 'attachment; filename="audio.wav"'
    return response