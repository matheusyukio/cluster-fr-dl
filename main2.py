import numpy as np # linear algebra
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# Pooling layers
from tensorflow.keras.layers import MaxPooling2D
# flatten layers into single vector
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, LocallyConnected2D, Input
from tensorflow.keras.layers import ZeroPadding2D
#import tensorflow_addons as tfa

# imports
import os
import gc
# visualizazao
import matplotlib.pyplot as plt
# Mover arquivos
from datetime import datetime
# CNN
from tensorflow.keras import optimizers

from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Model

from sklearn.model_selection import StratifiedKFold
import datetime
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

#diretorio onde fica a pasta do dataset lfw-dataset
DATA_PATH = '/mnt/c/Users/matheus.kumano/Desktop/Docker/notebooks/'

def get_data_transform():
    #base de dados
    #lfw_allnames = pd.read_csv("../input/lfw-dataset/lfw_allnames.csv")
    lfw_allnames = pd.read_csv(DATA_PATH+"lfw-dataset/lfw_allnames.csv")
    #Monta um DF como caminho para cada imagem da pessoa
    #Por exemplo se há 55 imagens da mesma pessoa é montado o DF dos caminhos para as 55 imagens dela
    # shape data frame so there is a row per image, matched to relevant jpg file
    image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
    image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
    image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
    image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
    image_paths = image_paths.drop("images",1)
    return image_paths

def get_min_img(image_paths,min_img):
    ind_counts = image_paths.groupby('name').count().image_path
    ind_counts[ind_counts >= min_img]
    image_list = []
    for img in ind_counts[ind_counts >= min_img].iteritems():
        image_list.append(img[0])
    return image_list

def mount_data(pd_df, min_img, sample_size):
    person_list = get_min_img(pd_df,min_img)
    total_filtered = pd_df[pd_df['name'].isin(person_list)]
    sample_list = []
    for img in person_list:
        sample_list.append(total_filtered[total_filtered.name==img].sample(sample_size))
    return pd.concat(sample_list)

def get_mounted_data(min_img, sample_size):
    image_paths = get_data_transform()
    return mount_data(image_paths, min_img, sample_size)
    
def dataTrainAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )
def dataHoldOutAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        validation_split=0.3
    )

def get_model(model_name, num_classes):
    if model_name == "create_new_model":
        return create_new_model(num_classes)
    elif model_name == "AlexNet":
        return AlexNet(num_classes)
    elif model_name == "LeNet5":
        return LeNet5(num_classes)
    elif model_name == "VGG16":
        return VGG16(num_classes)
    elif model_name == "ResNet50":
        return ResNet50(num_classes)
    elif model_name == "InceptionV3":
        return InceptionV3(num_classes)
    elif model_name == "DeepFace":
        return DeepFace(num_classes)
    elif model_name == "VGGFace":
        return VGGFace(num_classes)

def get_model_name(name,k,batch):
    return 'model_'+name+'_'+str(k)+'_'+str(batch)+'.h5'

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

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

def VGG16(num_classes):
    #input_tensor = Input(shape=(250, 250, 3))
    base_model = tf.keras.applications.VGG16(input_shape=(250, 250, 3),
                                               include_top=False,
                                               weights='imagenet')
    #base_model = VGG16(input_shape=(250, 250, 3), weights='imagenet',include_top=False)
    #x = base_model.output
    base_model.trainable = False
    
    
    
    #model = Model(inputs=model.inputs, outputs=predictions)
    # add new classifier layers
    flat1 = Flatten()
    class1 = Dense(1024, activation='relu')
    #output = Dense(10, activation='softmax')(class1)
    # define new model
    predictions = Dense(num_classes, activation='softmax')
    
    #for layer in base_model.layers: 
    #    layer.trainable = False
    model = tf.keras.Sequential([base_model, flat1,class1, predictions])
        
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
    #model.add(Activation('softmax'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def write_metrics(filename, y_pred, classes, nomes_classes):
	average_type = ['micro','macro']#,'samples'
	file = open('write_metrics '+filename, 'a+')
	for i in average_type:
	    score = f1_score(classes, y_pred, average=i)
	    precision = precision_score(classes, y_pred, average=i)
	    recall = recall_score(classes, y_pred, average=i)

	    file.write(i + ' f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
	    file.write('\n')

	file.write('\n')
	file.write("Accuracy: %f" % accuracy_score(classes, y_pred))
	file.write('\n')
	file.write('Confusion Matrix')
	file.write('\n')
	cm = confusion_matrix(classes, y_pred)
	file.write('cm')
	file.write('\n')
	file.write(str(cm))
	file.write('\n')
	file.write(classification_report(classes, y_pred, target_names=nomes_classes))
	file.write('\n')
	file.close()

def write_results(filename, acc, loss, history):
    max_acc_val = []
    VALIDATION_ACCURACY = acc
    VALIDATION_LOSS = loss
    HISTORY = history
    file = open('write_results '+filename, 'a+')
    file.write('VALIDATION_ACCURACY \n')
    file.write(str(VALIDATION_ACCURACY))
    file.write('\n')
    file.write('VALIDATION_ACCURACY mean\n')
    file.write(str(np.mean(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('VALIDATION_ACCURACY std\n')
    file.write(str(np.std(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('\n')
    file.write('VALIDATION_LOSS \n')
    file.write(str(VALIDATION_LOSS))
    file.write('\n\n')
    for hist in range(len(HISTORY)):
        file.write('VALIDATION_ACCURACY HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_ACCURACY[hist]))
        file.write('\n')
        file.write('MAX ACC ' + str(max(HISTORY[hist].history['val_acc'])) + ' \n')
        max_acc_val.append(max(HISTORY[hist].history['val_acc']))
        file.write('ACC MAX VALIDATION ARRAY ' + str(max_acc_val) + '\n')
        file.write('ACC MEAN VALIDATION ')
        file.write(str(np.mean(max_acc_val)))
        file.write('\n')
        if len(max_acc_val) > 1:
            file.write('ACC STD VALIDATION ' + str(np.std(max_acc_val)) + '\n')
        file.write('\n')
        file.write('VALIDATION_LOSS HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_LOSS[hist]))
        file.write('\n')
        file.write('HISTORY ' + str(hist) + ' \n')
        file.write('EPOCHS ' + str(len(HISTORY[hist].history['loss'])) + ' \n')
        file.write(str(HISTORY[hist].history))
        file.write('\n\n')
    file.close()
    plot_train_test_loss(HISTORY[hist].history, filename)

def plot_train_test_loss(history, filename):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(filename + ' modelo acc')
    plt.ylabel('acc')
    plt.xlabel('epoca')
    plt.legend(['treino', 'validacao'], loc='upper left')
    #plt.show()
    plt.savefig(filename + 'ACC.png')
    plt.close()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(filename +  ' modelo loss')
    plt.ylabel('loss')
    plt.xlabel('epoca')
    plt.legend(['treino', 'validacao'], loc='upper left')
    #plt.show()
    plt.savefig(filename + 'LOSS.png')
    plt.close()

def run_k_fold(multi_data, X, Y, CLASSES, epoch, MODEL, BATCH_SIZE, num_folds):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    HISTORY = []
    MODEL_NAME = MODEL
    FOLDS = num_folds
    EPOCHS = epoch
    save_dir = os.path.join(os.getcwd())
    VERBOSE = 1

    skf = StratifiedKFold(n_splits=FOLDS, random_state=7, shuffle=True)

    fold_var = 1
    for train_index, val_index in skf.split(X, Y):
        print("=======EPOCHS ", EPOCHS, " Start--k: ", fold_var)

        training_data = multi_data.iloc[train_index]
        validation_data = multi_data.iloc[val_index]

        print(training_data.shape)
        print(validation_data.shape)
        '''
        train_data_generator = dataTrainAugmentation().flow_from_directory(
            # training_data,
            directory=os.path.join(os.getcwd(), 'new/working/training_data_'+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var)+'/'),
            target_size=(250, 250),
            # x_col = "image_path", y_col = "name",
            batch_size=BATCH_SIZE,
            #subset="training",
            class_mode="categorical",
            shuffle=True)

        valid_data_generator = dataTrainAugmentation().flow_from_directory(
            # training_data,
            directory=os.path.join(os.getcwd(), 'new/working/validation_data_'+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var)+'/'),
            target_size=(250, 250),
            # x_col = "image_path", y_col = "name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            #subset="validation",
            shuffle=True)
            ../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/
        '''
        # flow_from_dataframe
        train_data_generator = dataTrainAugmentation().flow_from_dataframe(
            dataframe=training_data,
            directory=os.path.join(DATA_PATH+'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
            target_size=(250, 250),
            x_col="image_path", y_col="name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True)

        valid_data_generator = dataTrainAugmentation().flow_from_dataframe(
            dataframe=validation_data,
            directory=os.path.join(DATA_PATH+'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
            target_size=(250, 250),
            x_col="image_path", y_col="name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True)
        model = get_model(MODEL, CLASSES)
        # rmsprop = RMSprop(lr=1e-3, decay=1e-6)
        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+'/'+ get_model_name(MODEL_NAME, fold_var, BATCH_SIZE),monitor='val_acc', verbose=VERBOSE, save_best_only=True,mode='max')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE, patience=400)

        callbacks_list = [checkpoint, earlystopping]

        history = model.fit(train_data_generator,
                            epochs=EPOCHS,
                            steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
                            callbacks=callbacks_list,
                            validation_data=valid_data_generator,
                            validation_steps=valid_data_generator.n // valid_data_generator.batch_size,
                            verbose=VERBOSE,
                            #GPU Test luisss
                            max_queue_size=BATCH_SIZE,                # maximum size for the generator queue
                            workers=12,                        # maximum number of processes to spin up when using process-based threading
                            use_multiprocessing=False
                            )

        HISTORY.append(history)

        # LOAD BEST MODEL to evaluate the performance of the model model_"+MODEL_NAME+"_"+str(fold_var)+".h5"
        model.load_weights(
            os.getcwd() + "/model_" + MODEL_NAME + "_" + str(fold_var) + '_' + str(BATCH_SIZE) + ".h5")

        results = model.evaluate(valid_data_generator)
        # results = model.evaluate_generator(valid_data_generator)
        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])

        write_results(
            get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + '.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)

        Y_pred = model.predict_generator(valid_data_generator, validation_data.shape[0]//BATCH_SIZE + 1)
        y_pred = np.argmax(Y_pred, axis=1)

        nomes_classes = []
        for i in pd.DataFrame(Y.groupby('name')['name'].nunique().reset_index(name="unique"))[
            'name']:  # Y.groupby('name').nunique()['name']:
            nomes_classes.append(str(i))
        cm = confusion_matrix(valid_data_generator.classes, y_pred)

        write_metrics(get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(BATCH_SIZE) + '.txt', y_pred, valid_data_generator.classes, nomes_classes)

        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            plt.figure(figsize=(CLASSES+10, CLASSES+10))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            #plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('Classe real')
            plt.xlabel('Classe predita')
            plt.savefig(get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + 'CM.png')
            plt.close()

        plot_confusion_matrix(cm, classes=nomes_classes, title='Matriz de Confusão')
        
        del history
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        fold_var += 1


def main():
    epoch = 500
    min_images_per_person = [30]  # [25,20]
    models = ["AlexNet","DeepFace"]#,"AlexNet","LeNet5"] #["LeNet5","DeepFace","AlexNet"]#["DeepFace",AlexNet","LeNet5"]
    num_folds = 5

    #aumentando o batch para 30 DeepFace conseguiu bons resultados, testar com outras
    batch_sizes = [30,60]  # [2,4,8]
    for min_per_person in min_images_per_person:
        for batch in batch_sizes:
            for model in models:
                multi_data = get_mounted_data(min_per_person, min_per_person)
                print("### run_k_fold ", " epoch ", epoch, " min_per_person ", min_per_person, " CLASSES ", multi_data[['name']].groupby('name').nunique().shape[0],"model ", model, " batch_size ", batch)
                run_k_fold(multi_data, multi_data[['image_path']], multi_data[['name']], multi_data[['name']].groupby('name').nunique().shape[0], epoch, model, batch, num_folds)

if __name__ == "__main__":
    main()