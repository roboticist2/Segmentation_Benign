import tensorflow as tf
import cv2
import numpy as np
import random
import os

from tensorflow.keras.models import load_model

model = load_model("segmentation_unet.h5")

def segmentation():

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    length = len(X_test)
    i = random.sample(range(length),1)

    #predictions = model.predict(np.expand_dims(X_test[i], axis=0))
    predictions = model.predict(X_test[i])


    max_value = np.max(predictions)
    predictions_normalized = predictions / max_value #0~1 정규화
    predictions_normalized = (predictions_normalized * 255).astype(np.uint8) #0~255 정규화


    threshold = 130  # 예측값을 이진화할 임계값
    predictions_normalized = (predictions_normalized > threshold).astype(np.uint8) # 경계 적용 및 0~1 적용
    predictions_normalized = (predictions_normalized * 255).astype(np.uint8) #0~255 재 정규화


    predictions_normalized = np.reshape(predictions_normalized, (128, 128)) # (1,128,128,3) -> (128,128,3)

    """
    #sementation_image= np.squeeze(mask, axis=-0)
    segmentation_image= np.squeeze(predictions_normalized, axis=-0)
    """

    # 예측 결과 출력
    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 옵션을 사용하여 크기 조절 가능한 창으로 설정
    cv2.resizeWindow("Segmentation", 500, 500)  # 창 크기를 800x600 픽셀로 조절
    cv2.imshow("Segmentation", predictions_normalized)

    #print(np.shape(predictions_normalized))

    np.set_printoptions(threshold=np.inf)
    i = i[0]

    cv2.namedWindow("Original_image", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 옵션을 사용하여 크기 조절 가능한 창으로 설정
    cv2.resizeWindow("Original_image", 500, 500)  # 창 크기를 800x600 픽셀로 조절
    cv2.imshow("Original_image", X_test[i])

    cv2.namedWindow("GroundTruth", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 옵션을 사용하여 크기 조절 가능한 창으로 설정
    cv2.resizeWindow("GroundTruth", 500, 500)  # 창 크기를 800x600 픽셀로 조절
    cv2.imshow("GroundTruth", y_test[i])

#    cv2.imshow("Segmentation Mask", mask * 255)  # 이진화된 마스크를 0-255 범위로 변환하여 출력

    # 키 입력 대기 후 창 닫기
    cv2.waitKey(0)