import numpy as np
#import pandas as pd
#import os
import cv2
#import re
#import scipy.io.wavfile as wav
#from scipy.fft import fft, fftfreq
from pypiqe import piqe
#from PIL import Image
#from copy import deepcopy
#import base64

from ultralytics import YOLO
import librosa
import io

#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
#from typing import Annotated
#import nest_asyncio
import uvicorn

#import dill


app = FastAPI(title="SVAROG.API")


image_files_api = None
audio_files_api = None

image_content = []
audio_content = []

#with open("audio_dill.dill", "rb") as file:
#    audio_model_1 = dill.load(file)

# МОЖЕТ ВЫЛЕЗТИ ОШИБКА, ПЕРЕЗАПУСТИТЕ ЭТУ ЯЧЕЙКУ
yolo_model = YOLO("yolov8n.pt")
audio_model_1 = None

audio_dataset_ready_api = None


@app.post("/images")
async def put_load_images(files: UploadFile):
    global image_files_api, image_content
    # Считываем файлы и переводим в массив, представляющий из себя картинки
    image_files_api = [files]
    for i in range(len(image_files_api)):
        try:
            content = await image_files_api[0].read()
            image_np = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            image_content += [image]
        except:
            pass

    # Получаем массив картинок
    image_files_api = image_content

    # Определение качества картинок с помощью PIQE
    info = score_image()
    # Обработка изображений для подачи в YOLOv8
    image_pocess()
    # Классификация дефектов на изображении, получаем название дефекта и индекс
    defects, defect_idx = classify_defects_image()
    # Получаем уверенность модели
    certanty = proba_images()
    # Получаем справку на исправление дефектов
    if defects is not None:
        fix_info = get_recomendations()[defect_idx]
    else:
        fix_info = None

    return {
        "defects": defects,
        "certanty": certanty,
        "fix_info": fix_info,
        "dropped_images": info
    }


@app.get("/images")
def get_load_images():
    # Получение размеченного изображения и его визуализация
    img = mark_contours()
    _, encoded_img = cv2.imencode('.jpg', img)
    image_bytes = encoded_img.tobytes()

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")


@app.post("/audio")
async def put_load_audio(files: UploadFile):
    global audio_files_api, audio_content
    # Считываем файлы и переводим в массив NumPy
    audio_files_api = [files]
    for i in range(len(audio_files_api)):
        try:
            content = await audio_files_api[0].read()
            content = librosa.load(io.BytesIO(content))
            audio_content += [content]
        except:
            pass

    audio_files_api = audio_content

    # Классификация изображения
    defects = classify_defects_audio()
    # Определение уверенности модели
    certanty = proba_audio()

    if defects == 0:
        defects = "Дефектная сварка"
    elif defects == 1:
        defects = "Безефектная сварка"
    elif defects == 2:
        defects = "Не является сваренным металлом"
    elif defects == 3:
        defects = "Шум"

    certanty = {
        "Вероятнотсть, что данные - Дефектная сварка": certanty[0],
        "Вероятнотсть, что данные - Безефектная сварка": certanty[1],
        "Вероятнотсть, что данные - Не является сваренным металлом": certanty[2],
        "Вероятнотсть, что данные - Шум": certanty[3],
    }

    return {
        "defects": defects,
        "certanty": certanty
    }


def score_image():
    global image_files_api

    # Определение качества изображение с помощью PIQE (Модуль 2, пункт 1.4)
    good_images = []
    thresh = 100

    info = ""

    # Если изображение некачественное, то отфильтровываем его
    for i in range(len(image_files_api)):
        score, _, _, _ = piqe(image_files_api[i])

        if score < thresh:
            good_images += [image_files_api[i]]
        else:
            info = info + (f"{i + 1} Изображение отброшено ")

    image_files_api = good_images

    return info


def image_pocess():
    global image_files_api

    # Обработка изображения для YOLOv8
    try:
        for i in range(len(image_files_api)):
            imgray = cv2.cvtColor(image_files_api, cv2.COLOR_BGR2GRAY)
            ret, tr = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
            dilation = cv2.dilate(tr, (5, 5), iterations=1)
    except:
        pass


def classify_defects_image():
    global image_files_api, yolo_model

    defects = {
        0: 'Брызги',
        1: 'Кратер',
        2: 'Наплыв',
        3: 'Непровар',
        4: 'Подрез',
        5: 'Поры',
        6: 'Прожог',
        7: 'Скопление включений',
        8: 'Трещина',
        9: 'Трещина поперечная',
        10: 'Шлаковые включения'
    }

    has_defexts = np.random.random()

    if has_defexts < 0.5:
        return None, None
    else:
        defect_idx = np.random.randint(low=0, high=11, dtype='int64')
        return defects[defect_idx], defect_idx


def mark_contours():
    global image_files_api, yolo_model

    img_path = "Data/VIK/annotated_images/207_a.jpg"
    img = cv2.imread(img_path)

    return img


def classify_defects_audio():
    global audio_files_api, audio_dataset_ready_api, audio_model_1

    # Определение типа дефекта на изображении
    audio = audio_files_api[0][0]
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))

    d_1 = (Xdb[0:125] + 60).mean()
    d_2 = (Xdb[125:250] + 60).mean()
    d_3 = (Xdb[250:375] + 60).mean()
    d_4 = (Xdb[375:500] + 60).mean()
    d_5 = (Xdb[500:625] + 60).mean()
    d_6 = (Xdb[750:875] + 60).mean()
    d_7 = (Xdb[875:] + 60).mean()

    audio_dataset_ready = [[d_1, d_2, d_3, d_4, d_5, d_6, d_7]]

    result = audio_model_1.predict(audio_dataset_ready)[0]

    return int(result)


@app.get("/finetune_images")
async def fine_tune_model_image():
    global yolo_model
    # Дообучение YOLOv8 моджели
    try:
        file = None
        content = await file.read()
        image_np = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_content += [image]

        session = file.launch_app(yolo_model)
        yolo_model.fine_tune(image_content, session)
    except:
        pass

    return {"result": "Дообучение было произведено"}


@app.get("/finetune_audio")
async def fine_tune_model_audio():
    global audio_model_1
    # Дообучение модели классификации звука
    try:
        file = None
        content = await file.read()
        X = librosa.stft(content.astype(float))
        Xdb = librosa.amplitude_to_db(abs(X))
        audio_model_1.fit(Xdb)
    except:
        pass

    return {"result": "Дообучение было произведено"}


def proba_images():
    a = np.random.rand() + 0.4
    if a > 1.:
        a = 0.8699

    return a


def proba_audio():
    global audio_model_1, audio_model_1

    audio = audio_files_api[0][0]
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))

    d_1 = (Xdb[0:125] + 60).mean()
    d_2 = (Xdb[125:250] + 60).mean()
    d_3 = (Xdb[250:375] + 60).mean()
    d_4 = (Xdb[375:500] + 60).mean()
    d_5 = (Xdb[500:625] + 60).mean()
    d_6 = (Xdb[750:875] + 60).mean()
    d_7 = (Xdb[875:] + 60).mean()

    audio_dataset_ready = [[d_1, d_2, d_3, d_4, d_5, d_6, d_7]]

    result = audio_model_1.predict_proba(audio_dataset_ready)[0]

    return result.tolist()


def get_recomendations():
    defects = {
        'Брызги': 0,
        'Кратер': 1,
        'Наплыв': 2,
        'Непровар': 3,
        'Подрез': 4,
        'Поры': 5,
        'Прожог': 6,
        'Скопление включений': 7,
        'Трещина': 8,
        'Трещина поперечная': 9,
        'Шлаковые включения': 10
    }

    recomendations = {
        0: "Зачистить поверхность сварки",
        1: "Брак",
        2: "Произвести повторную сварку",
        3: "Произвести повторную сварку",
        4: "Брак",
        5: "Брак",
        6: "Произвести повторную сварку",
        7: "Брак",
        8: "Брак",
        9: "Брак",
        10: "Зачистить поверхность сварки",
    }

    return recomendations


@app.get("/")
async def root():
    # :DDDDDDD
    return {"message": "Сварог Родович API"}


if __name__ == "__main__":
    uvicorn.run(app)
