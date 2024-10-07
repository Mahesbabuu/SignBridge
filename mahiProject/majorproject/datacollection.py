import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands= 1)
offset = 20
imgSize = 300
counter = 0

folder = "C:\\Users\\Mahesh1234\\OneDrive\\Desktop\\major project\\Dataset\\I Love You"

while True :
    success , img = cap.read()
    hands , img = detector.findHands(img)
    if hands :
        hand = hands[0]
        x,y,w,h = hand ['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y-offset : y + h + offset , x-offset : x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            if imgCrop is not None and imgCrop.size != 0:
                imgResize = cv2.resize(imgCrop , (wCal , imgSize))
                if imgResize is not None:
                    imgResizeShape = imgResize.shape
                    # Proceed with further processing of imgResize
                else:
                    print("Error: Resized image is empty.")
            else:
                print("Error: Cropped image is empty.")
            #imgResize = cv2.resize(imgCrop ,(wCal , imgSize))
            print(imgCrop.shape)
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) /2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            print("wCal:", wCal)
            print("imgSize:", imgSize)

        else :
            k = imgSize / w
            hCal = math.ceil(k*h)
            if imgCrop is not None and imgCrop.size != 0:
                imgResize = cv2.resize(imgCrop , (imgSize , hCal))
                if imgResize is not None:
                    imgResizeShape = imgResize.shape
                    # Proceed with further processing of imgResize
                else:
                    print("Error: Resized image is empty.")
            else:
                print("Error: Cropped image is empty.")
            #imgResize = cv2.resize(imgCrop ,(imgSize , hCal))
            print(imgCrop.shape)
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) /2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            print("wCal:", wCal)
            print("imgSize:", imgSize)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)
    
    cv2.imshow('Image' , img)
    key = cv2.waitKey(1)
    if key == ord('s') :
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
        
