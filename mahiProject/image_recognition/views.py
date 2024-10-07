from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import subprocess
import requests

import os

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
import math
def hello_world(request):
    return HttpResponse("Hello, World!")
# def run_ml_script(request):
#     try:
#         # Initialize the hand detector and classifier
#         detector = HandDetector(maxHands=1)
#         model_path = "F:/user 2 from c drive/Downloads/backend_mahi/keras_model.h5"
#         labels_path = "F:/user 2 from c drive/Downloads/backend_mahi/labels.txt"
#         classifier = Classifier(model_path, labels_path)

#         # Initialize video capture
#         cap = cv2.VideoCapture(0)

#         # Set up variables for processing
#         offset = 20
#         imgSize = 300
#         labels = ["Hello", "Thank You", "Yes", "No", "Please", "Like", "Dislike", "Call me",
#                   "Peace", "Food", "Nice", "Good luck", "Talk", "Look", "I Love You"]

#         while True:
#             # Read frame from the camera
#             success, img = cap.read()
#             imgOutput = img.copy()

#             # Detect hands in the frame
#             hands, img = detector.findHands(img)

#             if hands:
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']

#                 imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#                 # Crop and resize hand region
#                 imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#                 imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

#                 # Get prediction from classifier
#                 prediction, index = classifier.getPrediction(imgResize, draw=False)

#                 # Overlay prediction label on the frame
#                 cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50),
#                               (0, 255, 0), cv2.FILLED)
#                 cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
#                 cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

#                 # Display images
#                 cv2.imshow('ImageCrop', imgCrop)
#                 cv2.imshow('ImageWhite', imgWhite)

#             cv2.imshow('Image', imgOutput)
#             cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#             # Check for key press to exit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # Release video capture and close OpenCV windows
#         cap.release()
#         cv2.destroyAllWindows()

#         # Example response
#     #     return JsonResponse({'success': True, 'output': 'Machine learning script completed'})

#     # except Exception as e:
#     #     return JsonResponse({'success': False, 'error': str(e)})
#         context = {'success': True, 'output': 'Machine learning script completed'}
#         return render(request, 'index.html', context)

#     except Exception as e:
#         context = {'success': False, 'error': str(e)}
#         return render(request, 'index.html', context)


def run_ml_script(request):
    try:
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        model_path = "F:/user 2 from c drive/Downloads/Mahi_web/mahiProject/majorproject/keras_model.h5"
        labels_path = "F:/user 2 from c drive/Downloads/Mahi_web/mahiProject/majorproject/labels.txt"  # Absolute path

        classifier = Classifier(model_path, labels_path)
        offset = 20
        imgSize = 300
        labels = ["Hello", "Thank You", "Yes", "No", "Please", "Like", "Dislike", "Call me", "Peace", "Food", "Nice", "Good luck", "Talk", "Look", "I Love You"]

        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
                cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

            cv2.imshow('Image', imgOutput)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break  # Exit the loop if 'q' is pressed

        # Release video capture and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        context = {'success': True, 'output': 'Machine learning script completed'}
        return render(request, 'index.html', context)

    except Exception as e:
        context = {'success': False, 'error': str(e)}
        return render(request, 'index.html', context)


def index(request):
    return render(request, 'index.html')
