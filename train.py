import argparse
import os
from typing import Any

from PIL import Image

from common import load_image_labels, save_model, save_model_ensemble,load_images

import cv2
import pandas as pd
import os
import random
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import pickle
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping



########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args


'''def load_train_resources(resource_dir: str = 'resources') -> Any:
    """
    Load any resources (i.e. pre-trained models, data files, etc) here.
    Make sure to submit the resources required for your algorithms in the sub-folder 'resources'
    :param resource_dir: the relative directory from train.py where resources are kept.
    :return: TBD
    """
    raise RuntimeError(
        "load_train_resources() not implement. If you have no pre-trained models you can comment this out.")
'''
#creating stats file
def log_stats(output_dir, model_name, stats):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{model_name}_stats.json")
    with open(log_file, "w") as f:
        json.dump(stats, f, indent=4)

#Perform data augmentation to create more training data and create a new csv file with the original and augmented data
def augmentation(images,labels,csv_file,target_column_name,data_folder):
    for index, row in csv_file.iterrows():
        img = cv2.imread(images[index])
        label = labels[index]
        filename = row['Filename']
        
        # Initialize list to store augmented images
        augmented_images = []

        # Rotation
        angle = random.uniform(-30, 30)
        rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated_image)

        # Translation
        x_translation = random.randint(-50, 50)
        y_translation = random.randint(-50, 50)
        translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        translated_image = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
        augmented_images.append(translated_image)

        # Zoom
        scale_factor = random.uniform(0.8, 1.2)
        resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        augmented_images.append(resized_image)

        # Flip
        flip_code = random.choice([-1, 0, 1])
        flipped_image = cv2.flip(img, flip_code)
        augmented_images.append(flipped_image)

        # Brightness
        brightness_factor = random.uniform(0.5, 1.5)
        brightness_image = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        augmented_images.append(brightness_image)

        # Contrast
        contrast_factor = random.uniform(0.5, 1.5)
        contrast_image = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=50 * (1 - contrast_factor))
        augmented_images.append(contrast_image)

        # Crop
        x_start = random.randint(0, img.shape[1] // 2)
        y_start = random.randint(0, img.shape[0] // 2)
        cropped_image = img[y_start:y_start + img.shape[0] // 2, x_start:x_start + img.shape[1] // 2]
        augmented_images.append(cropped_image)

        # Shear
        shear_factor = random.uniform(-0.2, 0.2)
        shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])
        sheared_image = cv2.warpAffine(img, shear_matrix, (img.shape[1], img.shape[0]))
        augmented_images.append(sheared_image)

        # Normalize (skip if already normalized)
        normalized_image = img.astype(np.float32) / 255.0

        # Gaussian Noise
        mean = 0
        stddev = 25
        noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
        noisy_image = cv2.add(img, noise)
        augmented_images.append(noisy_image)

        # Color Channel Shifting
        b, g, r = cv2.split(img)
        shifted_image = cv2.merge([r, b, g])
        augmented_images.append(shifted_image)

        # Histogram Equalization
        equalized_image = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        augmented_images.append(equalized_image)

        # Saturation (HSV space)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * 0.5
        desaturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        augmented_images.append(desaturated_image)

        # Hue (HSV space)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 10) % 180
        hue_shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        augmented_images.append(hue_shifted_image)

        # Blur
        blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
        augmented_images.append(blurred_image)

        # Sharpen
        kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_image = cv2.filter2D(img, -1, kernel_sharpening)
        augmented_images.append(sharpened_image)

        # Scale (resize to a fixed size)
        resized_image = cv2.resize(img, (500, 500))  # Example: resizing to 500x500
        augmented_images.append(resized_image)

        # Save augmented images and update CSV
        for i, augmented_image in enumerate(augmented_images):
            augmented_output_path = os.path.join(data_folder, f"{os.path.splitext(filename)[0]}_{i}_{label}.jpg")
            cv2.imwrite(augmented_output_path, augmented_image)
            print(f"Augmented image saved: {augmented_output_path}")

            # Update CSV with new augmented image information
            new_row = {'Filename': os.path.basename(augmented_output_path), target_column_name: label}
            csv_file = pd.concat([csv_file, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated CSV
    updated_csv_path = os.path.join(data_folder, f"{target_column_name}.csv")
    csv_file.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved: {updated_csv_path}")
    return updated_csv_path

def extract_features(updated_csv_path,data_folder):
    vgg16_model = VGG16(weights='imagenet', include_top=False)

    vgg16_features = []
    csv_file = pd.read_csv(updated_csv_path)
    for index, row in csv_file.iterrows():
        img_name = row['Filename']
        #print(row['Filename'])
        img_path = os.path.join(data_folder, img_name)
        img = image.load_img(img_path, target_size=(224, 224,3))

        # Convert image to array
        x = image.img_to_array(img)

        # Reshape the image
        x = np.expand_dims(x, axis=0)

        # Preprocess the input based on the model
        x_vgg16 = vgg16_preprocess_input(x.copy())

        # Extract features
        features_vgg16 = vgg16_model.predict(x_vgg16)

        # Append features to the list
        vgg16_features.append(features_vgg16)

    # Convert the features list to numpy array
    vgg16_features = np.array(vgg16_features)
    
    vgg16_features = np.reshape(vgg16_features, (len(vgg16_features), 7 * 7 * 512))
    return vgg16_features

def vgg16_model(vgg16_features,labels,X_train,X_test,y_train,y_test):
    model = models.Sequential()
    model.add(layers.Dense(2048, activation='relu', input_dim=7 * 7 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', input_dim=7 * 7 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
    history = model.fit(X_train, y_train,
                    epochs=25,
                    batch_size=16,
                    validation_data=(X_test, y_test))
    return model,history

def random_forest_model(vgg16_features,labels):
    X_for_RF = vgg16_features #This is our X input to RF

    RF_model = RandomForestClassifier(n_estimators = 5000, random_state = 42)

    # Train the model on training data
    RF_model.fit(X_for_RF, labels) #For sklearn no one hot encoding
    return RF_model

def SVM_model(vgg16_features,labels):
    svm_clf = svm.SVC(kernel='linear')
    # Train SVM classifier
    labels = np.argmax(labels, axis=1)
    svm_clf.fit(vgg16_features, labels)
    return svm_clf

def Knn_model(vgg16_features,labels):
    knn = KNeighborsClassifier(n_neighbors=3,metric='cosine')
    knn.fit(vgg16_features,labels)
    return knn

def decision_tree_model(vgg16_features,labels):
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(vgg16_features,labels)
    return dt_classifier

def dense_net(X_train,X_val,y_train,y_val):
    # Load the pre-trained DenseNet model
    Load_pretrained_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in Load_pretrained_model.layers:
        layer.trainable = False

    # Add your own layers after second last layer
    x = GlobalAveragePooling2D()(Load_pretrained_model.get_layer('conv5_block16_concat').output)
    x = Dropout(0.5)(x)  # Add dropout layer
    outputs = Dense(1, activation='sigmoid')(x)  # Add output layer for binary classification
    model_pretrained = Model(inputs=Load_pretrained_model.input, outputs=outputs)
    model_pretrained.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model_pretrained.fit(X_train, y_train, epochs=150, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model_pretrained,history



def train(images: [Image], labels: [str], output_dir: str,data_folder: str,csv_file,target_column_name: str,labels_file_path: str,updated_csv_path) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.
    
    #VGG model for feature extraction
    vgg16_features = extract_features(updated_csv_path,data_folder)
    
    #One Hot encoding the labels
    one_hot_labels = csv_file[target_column_name].tolist()
    label_encoder = LabelEncoder()
    one_hot_labels = label_encoder.fit_transform(one_hot_labels)
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    one_hot_labels = to_categorical(one_hot_labels)

    X_train,X_test,y_train,y_test = train_test_split(vgg16_features, one_hot_labels, test_size=0.2, random_state=42)
    model_vgg16,history_vgg = vgg16_model(vgg16_features,one_hot_labels,X_train,X_test,y_train,y_test)
    model_random_forest= random_forest_model(vgg16_features,one_hot_labels)
    model_svm = SVM_model(vgg16_features,one_hot_labels)
    model_knn = Knn_model(vgg16_features,one_hot_labels)
    model_dt = decision_tree_model(vgg16_features,one_hot_labels)

    #DenseNet model
    images = load_images(images)
    resized_images=[]
    for e in images:
        new_image = cv2.resize(e,(224,224))
        resized_images.append(new_image)
    X_train, X_test, y_train, y_test = train_test_split(resized_images, labels, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array([0 if label == 'No' else 1 for label in y_train] )
    y_test = np.array([0 if label == 'No' else 1 for label in y_test ])
    model_dense_net,history_dense_net = dense_net(X_train,X_test,y_train,y_test)

    log_stats(output_dir, 'vgg16', history_vgg.history)
    log_stats(output_dir, 'dense_net', history_dense_net.history)
    return model_vgg16,model_random_forest,model_svm,model_knn,model_dt,model_dense_net


def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """

    # load pre-trained models or resources at this stage.
    #load_train_resources()

    # load label file
    labels_file_path = os.path.join(train_input_dir, train_labels_file_name)
    df_labels = load_image_labels(labels_file_path)

    # load in images and labels
    train_images = []
    train_labels = []
    # Now iterate through every record and load in the image data files
    # Given the small number of data samples, iterrows is not a big performance issue.
    for index, row in df_labels.iterrows():
        try:
            filename = row['Filename']
            label = row[target_column_name]

            print(f"Loading image file: {filename}")
            image_file_path = os.path.join(train_input_dir, filename)
            #image = load_single_image(image_file_path)

            train_labels.append(label)
            train_images.append(image_file_path)
        except Exception as ex:
            print(f"Error loading {index}: {filename} due to {ex}")
    print(f"Loaded {len(train_labels)} training images and labels")

    # Create the output directory and don't error if it already exists.
    os.makedirs(train_output_dir, exist_ok=True)
    updated_csv_path = augmentation(train_images,train_labels,df_labels,target_column_name,train_input_dir)
    df_labels = load_image_labels(updated_csv_path)
    for index, row in df_labels.iterrows():
        try:
            filename = row['Filename']
            label = row[target_column_name]

            print(f"Loading image file: {filename}")
            image_file_path = os.path.join(train_input_dir, filename)
            #image = load_single_image(image_file_path)

            train_labels.append(label)
            train_images.append(image_file_path)
        except Exception as ex:
            print(f"Error loading {index}: {filename} due to {ex}")
    # train a model for this task
    model_vgg16,model_rf,model_svm,model_knn,model_dt,model_dense_net = train(train_images, train_labels, train_output_dir,train_input_dir,df_labels,target_column_name,labels_file_path,updated_csv_path)

    # save model
    save_model(model_vgg16, target_column_name, train_output_dir,"vgg16")
    save_model_ensemble(model_rf, target_column_name, train_output_dir, "rf")
    save_model_ensemble(model_svm, target_column_name, train_output_dir, "svm")
    save_model_ensemble(model_knn, target_column_name, train_output_dir, "knn")
    save_model_ensemble(model_dt, target_column_name, train_output_dir, "dt")
    save_model(model_dense_net, target_column_name, train_output_dir, "dense_net")

if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)

########################################################################################################################
