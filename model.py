from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.losses import categorical_crossentropy
import cv2

from data import data_processing
# U-Net 모델 정의

def modeling():

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_processing()

    def unet_model(input_size=(128, 128, 3)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)

        up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
        conv4 = Conv2D(128, 3, activation="relu", padding="same")(up4)
        conv4 = Conv2D(128, 3, activation="relu", padding="same")(conv4)

        up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
        conv5 = Conv2D(64, 3, activation="relu", padding="same")(up5)
        conv5 = Conv2D(64, 3, activation="relu", padding="same")(conv5)

        outputs = Conv2D(1, 1, activation="sigmoid")(conv5)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    # 모델 생성
    model = unet_model()

    # 모델 컴파일
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 모델 학습
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=10, epochs=5, verbose=1)

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 모델 저장
    model.save("segmentation_unet.h5")