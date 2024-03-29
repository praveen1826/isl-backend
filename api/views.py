from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
# import tensorflow_hub as hub
import keras
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
from keras import preprocessing
from keras.models import load_model
from keras.activations import softmax
from sklearn.preprocessing import OneHotEncoder
import os
import h5py
from django.views.decorators.csrf import csrf_exempt


model = keras.models.load_model('api/mlModel/model.h5')
# shape = ((50, 50, 3))
# model = tf.keras.Sequential([hub.KerasLayer(model, input_shape=shape)])

modelV2 = keras.models.load_model('api/mlModel/model2.0.h5')

modelV3 = keras.models.load_model('api/mlModel/model_cnn.keras')

# modelV4 = keras.models.load_model('api/mlModel/MobileNet.keras')

input_layer = keras.layers.Input(shape=(224, 224, 3))
x = keras.layers.TFSMLayer('api/mlModel/MobileNetFT',
                           call_endpoint='serving_default')(input_layer)
modelV4 = keras.models.Model(inputs=input_layer, outputs=x)


@csrf_exempt
def predict(request):
    if request.method == 'POST':

        # image = Image.open("api/mlModel/0.jpg")

        # Get the image from the request
        print(request.FILES['image'])
        image = Image.open(request.FILES['image'])

        # Preprocess the image

        test_image = image.resize((50, 50))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                       'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        # Make a prediction
        predictions = model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        print(image_class)

        return JsonResponse({'prediction': image_class})

    else:
        return render(request, 'predict.html')


# Create your views here.

def form_view(request):
    return render(request, 'predict.html')


@csrf_exempt
def predictV2(request):
    if request.method == 'POST':

        # image = Image.open("api/mlModel/0.jpg")

        # Get the image from the request
        print(request.FILES['image'])
        # image1 = Image.open(request.FILES['image'])
        # img = cv.imread(Image.open(request.FILES['image']))
        img = cv.imdecode(np.fromstring(
            request.FILES['image'].read(), np.uint8), cv.IMREAD_UNCHANGED)
        # Preprocess the image

        resized_img = cv.resize(img, (250, 250), interpolation=cv.INTER_CUBIC)
        resized_img.shape
        # plt.imshow(resized_img)
        img = resized_img
        pred = modelV2.predict(x=np.array(
            img).reshape(-1, 250, 250, 3)).flatten()

        enc = OneHotEncoder()
        enc.fit([['6'],
                ['K'],
                ['L'],
                ['R'],
                ['V'],
                ['3'],
                ['F'],
                ['M'],
                ['J'],
                ['0'],
                ['9'],
                ['U'],
                ['8'],
                ['P'],
                ['W'],
                ['Q'],
                ['N'],
                ['E'],
                ['Y'],
                ['H'],
                ['1'],
                ['X'],
                ['C'],
                ['G'],
                ['5'],
                ['O'],
                ['S'],
                ['B'],
                ['2'],
                ['7'],
                ['D'],
                ['T'],
                ['4'],
                ['I'],
                ['A'],
                ['Z']])
        out = enc.inverse_transform(pred.reshape(1, -1))
        print(out[0][0])

        return JsonResponse({'prediction': out[0][0]})

    else:
        return render(request, 'predict.html')


@csrf_exempt
def predictV3(request):
    if request.method == 'POST':

        # image = Image.open("api/mlModel/0.jpg")

        # Get the image from the request
        print(request.FILES['image'])
        image = Image.open(request.FILES['image'])

        # Preprocess the image

        test_image = preprocessing.image.img_to_array(image)
        test_image = tf.expand_dims(test_image, 0)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                       'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Make a prediction
        predictions = modelV3.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        print(image_class)

        return JsonResponse({'prediction': image_class})

    else:
        return render(request, 'predict.html')


@csrf_exempt
def predictV4(request):
    if request.method == 'POST':

        # image = Image.open("api/mlModel/0.jpg")

        # Get the image from the request
        print(request.FILES['image'])
        image = Image.open(request.FILES['image'])

        # Preprocess the image

        image = image.resize((224, 224))

        test_image = preprocessing.image.img_to_array(image)
        test_image = tf.expand_dims(test_image, 0)
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                       'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Make a prediction
        predictions = modelV4.predict(test_image)
        # print(predictions)
        scores = tf.nn.softmax(predictions['dense_1'])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        print(image_class)

        return JsonResponse({'prediction': image_class})

    else:
        return render(request, 'predict.html')
