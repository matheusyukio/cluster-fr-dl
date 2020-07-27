from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# Pooling layers
from tensorflow.keras.layers import MaxPooling2D
# flatten layers into single vector
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, LocallyConnected2D
from tensorflow.keras.layers import ZeroPadding2D

#import tensorflow_addons as tfa


def create_new_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = num_classes, activation = 'softmax'))
    return model

def DeepFace(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (11, 11), activation='relu', name='C1', input_shape=(250, 250, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    model.add(Conv2D(16, (9, 9), activation='relu', name='C3'))
    model.add(Conv2D(16, (9, 9), activation='relu', name='L4'))
    model.add(Conv2D(16, (7, 7), strides=2, activation='relu', name='L5'))
    model.add(Conv2D(16, (5, 5), activation='relu', name='L6'))
    model.add(Flatten(name='F0'))
    model.add(Dense(4096, activation='relu', name='F7'))
    #taxa de dropout, ajustar aumentar e testar
    # tirar imagem media
    model.add(Dropout(rate=0.5, name='D0'))
    model.add(Dense(num_classes, activation='softmax', name='F8'))
    return model

def LeNet5(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(250, 250, 3)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def AlexNet(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     input_shape=(250, 250, 3), strides=(4, 4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(250 * 250 * 3,), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def VGGFace(num_classes):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(250,250, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1)))
    model.add(Flatten())
    #2)PEGAR O MAPA DE ATIVACAO GERADO -> VETOR  DO MELHOR PARAMETRO DURANTE O TREINAMENTO
    #VERIFICAR SE FEATURES EXTRAÍDAS ESTAO BOAS
    #model.add(Activation('softmax'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def VGGFaceHalf(num_classes):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(250,250, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(2048, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2048, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1311, (1, 1)))
    model.add(Flatten())
    #model.add(Activation('softmax'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model