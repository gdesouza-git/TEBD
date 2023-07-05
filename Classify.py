import json
import random
import time
import requests
from DBInsert import insert
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import urllib.request
import os

def classificar(audio):
    print("LOG:")
    urllib.request.urlretrieve(audio, 'audio.wav')
    print("- Baixando audio")
    time.sleep(10) #garante que de tempo do audio ser baixado
    print(os.path.exists(audio))
    language = 'pt-BR'
    modelo = 'qanastek/XLMRoberta-Alexa-Intents-Classification'
    url = "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=" + language

    headers = {
       'Content-type': 'audio/wav;codec="audio/pcm";',
       'Ocp-Apim-Subscription-Key': '',
    }

    with open('audio.wav', 'rb') as payload:
        response = requests.request("POST", url, headers=headers, data=payload)
        print("- Transcrição realizada pelo Azure:")
        print(response.text)

    response = json.loads(response.text)

    model_name = modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    print("- Modelo carregado")

    res = classifier(response['DisplayText'])
    print("- Classificação concluída")
    id = random.randint(1, 1000000)

    result = {
       "id": id,
       "RecognitionStatus": response['RecognitionStatus'],
       "Intent": res[0]['label'],
       "TranscriptionText": response['DisplayText'],
       "Score": res[0]['score'],
       "Offset": response['Offset']
    }

    insert(result)
    #os.remove('audio.wav')
    print("- Arquivos residuais removidos")
    return str(result)
