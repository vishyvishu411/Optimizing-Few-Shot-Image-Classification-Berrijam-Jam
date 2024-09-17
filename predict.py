import argparse
import os
from typing import Any

import pandas as pd
from PIL import Image
import pickle
from common import load_model, load_predict_image_names,load_images,load_model_ensemble
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing import image
import cv2


########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args

def extract_features(img):
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    vgg16_features = []
    img = image.load_img(img, target_size=(224, 224,3))
    # Convert image to array
    x = image.img_to_array(img)

    # Reshape the image
    x = np.expand_dims(x, axis=0)

    # Preprocess the input based on the model)
    x_vgg16 = vgg16_preprocess_input(x.copy())

    # Extract features
    features_vgg16 = vgg16_model.predict(x_vgg16)
    vgg16_features = np.array(features_vgg16)

    vgg16_features = np.reshape(vgg16_features, (len(vgg16_features), 7 * 7 * 512))
    return vgg16_features

def to_binary(preds):
    return 1 if preds == "Yes" else 0
    

def predict(vgg16_model: Any, dense_net_model: Any, rf_model: Any, dt_model: Any,svm_model: Any, knn_model:Any, img:str,trained_model_dir) -> str:
    """
    Generate a prediction for a single image using the model, returning a label of 'Yes' or 'No'

    IMPORTANT: The return value should ONLY be either a 'Yes' or 'No' (Case sensitive)

    :param model: the model to use.
    :param image: the image file to predict.
    :return: the label ('Yes' or 'No)
    """
    predicted_label_vgg = []
    predicted_label_densenet = []
    predicted_label_rf = []
    predicted_label_svm = []
    predicted_label_dt = []
    predicted_label_knn = []
    predicted_label = ''
    
    with open(os.path.join(trained_model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    vgg16_features = extract_features(img)

    #Prediction for VGG16 with custom dense layers
    y_pred=vgg16_model.predict(vgg16_features)
    classes_x=np.argmax(y_pred,axis=1)
    predicted_label_vgg=label_encoder.inverse_transform(classes_x)[0]
    predicted_label_vgg_binary = to_binary(predicted_label_vgg)

    #Prediction using VGG16 with Random forest
    y_pred=rf_model.predict(vgg16_features)
    classes_x=np.argmax(y_pred,axis=1)
    predicted_label_rf=label_encoder.inverse_transform(classes_x)[0]
    predicted_label_rf_binary = to_binary(predicted_label_rf)

    #Prediction with VGG16 with SVM
    y_pred=svm_model.predict(vgg16_features)
    predicted_label_svm=label_encoder.inverse_transform(y_pred)[0]
    predicted_label_svm_binary = to_binary(predicted_label_svm)

    #Prediction with VGG16 with Decision Tree
    y_pred=dt_model.predict(vgg16_features)
    classes_x=np.argmax(y_pred,axis=1)
    predicted_label_dt=label_encoder.inverse_transform(classes_x)[0]
    predicted_label_dt_binary = to_binary(predicted_label_dt)

    #Prediction with VGG16 with KNN
    y_pred=knn_model.predict(vgg16_features)
    classes_x=np.argmax(y_pred,axis=1)
    predicted_label_knn=label_encoder.inverse_transform(classes_x)[0]
    predicted_label_knn_binary = to_binary(predicted_label_knn)
    
    #Prediction with Densenet
    image = load_images([img])
    new_image = []
    for i in image:
        new_image.append(cv2.resize(i, (224, 224)))
    img = np.array(new_image)
    y_pred=dense_net_model.predict(img)
    
    predicted_label_densenet = 'Yes' if y_pred > 0.5 else 'No'
    predicted_label_densenet_binary = to_binary(predicted_label_densenet)

    all_preds = np.vstack([
        predicted_label_vgg_binary,
        predicted_label_rf_binary,
        predicted_label_dt_binary,
        predicted_label_svm_binary,
        predicted_label_knn_binary,
        predicted_label_densenet_binary
    ])

    ensemble_preds = np.mean(all_preds, axis=0) > 0.5

    predicted_label = np.array(["Yes" if pred == 1 else "No" for pred in ensemble_preds])[0]
    print("Final Prediction:",predicted_label)

    return predicted_label


def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """

    # load pre-trained models or resources at this stage.
    vgg16_model = load_model(trained_model_dir, target_column_name,"vgg16")
    dense_net_model = load_model(trained_model_dir, target_column_name,"dense_net")
    rf_model = load_model_ensemble(trained_model_dir, target_column_name,"rf")
    svm_model = load_model_ensemble(trained_model_dir, target_column_name,"svm")
    dt_model = load_model_ensemble(trained_model_dir, target_column_name,"dt")
    knn_model = load_model_ensemble(trained_model_dir, target_column_name,"knn")

    # Load in the image list
    image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
    image_filenames = load_predict_image_names(image_list_file)

    # Iterate through the image list to generate predictions
    predictions = []
    for filename in image_filenames:
        try:
            image_path = os.path.join(predict_data_image_dir, filename)
            #image = load_single_image(image_path)
            label = predict(vgg16_model,dense_net_model,rf_model,dt_model,svm_model,knn_model,image_path,trained_model_dir)
            predictions.append(label)
        except Exception as ex:
            print(f"Error generating prediction for {filename} due to {ex}")
            predictions.append("Error")

    df_predictions = pd.DataFrame({'Filenames': image_filenames, target_column_name: predictions})

    # Finally, write out the predictions to CSV
    df_predictions.to_csv(predicts_output_csv, index=False)


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################