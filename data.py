import numpy as np
import cv2
import os
import random
import tensorflow as tf
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

data_dir = "MT_Small_Dataset/Benign/Fuzzy_Benign/"
ground_truth_dir = "MT_Small_Dataset/Benign/Ground_Truth_Benign/"
image_files = os.listdir(os.path.join(data_dir))
ground_truth_files = os.listdir(os.path.join(ground_truth_dir))

def data_processing() :

    images = []
    ground_truths = []

    for img_file, gt_file in zip(image_files, ground_truth_files):
        image = tf.keras.preprocessing.image.load_img(
            os.path.join(data_dir, img_file), target_size=(128, 128)
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        images.append(image)

        gt_image = tf.keras.preprocessing.image.load_img(
            os.path.join(ground_truth_dir, gt_file), target_size=(128, 128), color_mode="grayscale"
        )
        gt_image = tf.keras.preprocessing.image.img_to_array(gt_image)
        ground_truths.append(gt_image)

    np.set_printoptions(threshold=np.inf)
#    print(images[0])
#    print(ground_truths[0])

    images = np.array(images) / 255.0  # 이미지 정규화 0~1

    # 데이터 분할 비율 설정
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1

    # 데이터 개수
    total_samples = len(images)
    train_samples = int(train_ratio * total_samples)
    valid_samples = int(valid_ratio * total_samples)

    # 데이터 인덱스 섞기
    indices = np.random.permutation(total_samples)

    # 데이터 분할
    train_indices = indices[:train_samples]
    valid_indices = indices[train_samples:train_samples+valid_samples]
    test_indices = indices[train_samples+valid_samples:]

    # ground_truths 이진화
    binary_ground_truths = []

    #ground_truths 초기값 : object=2, background=1
    ground_truths = np.array(ground_truths) # ground_truth는 max값 정규화

    for gt_image in ground_truths:
        max_value = np.max(gt_image)
        gt_image = gt_image / max_value
        gt_image = (gt_image * 255).astype(np.uint8)

        threshold = 130
        gt_image = (gt_image > threshold).astype(np.uint8)
        binary_ground_truths.append(gt_image)
    
    binary_ground_truths = np.array(binary_ground_truths)*255

    X_train, y_train = images[train_indices], binary_ground_truths[train_indices]
    X_valid, y_valid = images[valid_indices], binary_ground_truths[valid_indices]
    X_test, y_test = images[test_indices], binary_ground_truths[test_indices]

    if os.path.exists("X_test.npy"):
        os.remove("X_test.npy")
    if os.path.exists("y_test.npy"):
        os.remove("y_test.npy")

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    """
    #테스트 출력
    length = len(X_test)
    i = random.sample(range(length),1)
    i = i[0]

    cv2.namedWindow("Original_image", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 옵션을 사용하여 크기 조절 가능한 창으로 설정
    cv2.resizeWindow("Original_image", 800, 600)  # 창 크기를 800x600 픽셀로 조절
    cv2.imshow("Original_image", X_test[i])

    cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 옵션을 사용하여 크기 조절 가능한 창으로 설정
    cv2.resizeWindow("Ground Truth", 800, 600)  # 창 크기를 800x600 픽셀로 조절
    cv2.imshow("Ground Truth", y_test[i])

    # 키 입력 대기 (아무 키나 누를 때까지 대기)
    cv2.waitKey(0)
    """

    return X_train, y_train, X_valid, y_valid, X_test, y_test