import imageio
import json
from pathlib import Path
import SimpleITK as sitk
from app import app
from keras.models import load_model
from scipy.ndimage import zoom
from functools import partial
import tensorflow as tf
import numpy as np
from keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
import cv2
import os
from multiprocessing.pool import ThreadPool
from config import Config
import segmentation_models as sm
from PIL import Image
from matplotlib import cm, colors
from matplotlib import pyplot as plt
K.set_image_data_format('channels_first')


def read_img(img_path):
    SIZE = app.config["IMAGE_SIZE"]
    img = Image.open(img_path).resize(size= (SIZE, SIZE)).convert(mode='L')
    return img

def preprocess(img):
    img_arr = []
    img_arr.append(img)
    img = np.array(img_arr)
    img = np.expand_dims(img, axis=-1)  
    img = tf.keras.utils.normalize(img, axis=1)
    return img

def postprocess(mask):
    kernel = np.ones((3, 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j, ...] = cv2.dilate(mask[i, j, ...], kernel, iterations=1)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'iou_score': metrics[0],
                      'f1-score': metrics[1]}
    try:
        with open(f'{model_file}.json', 'r') as json_file:
            model = model_from_json(json_file.read(), custom_objects=custom_objects)
        model.load_weights(f'{model_file}.h5')
        # return load_model(model_file, custom_objects=custom_objects)
        return model
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise ValueError(str(error)+ '\n\nError loading the model')


def predict(img):
    global model
    pred = model.predict(img)
    pred = np.argmax(pred, axis=3)[0,:,:]
    return pred


def make_png(file):
    global model#, graph
    pred = []
    if 'model' not in globals():
        model = load_old_model(app.config['MODEL'])

    img_org = np.load(f'{app.config["UPLOAD_FOLDER"]}/npy/{file}')
    img = preprocess(img_org)
    pred = predict(img)

    ## PLOT IMAGE AND PRED

    zones = app.config['NO_ZONES']
    colours = cm.get_cmap('hsv', zones)
    cmap = colours(np.linspace(0,1,zones))
    cmap[0,:] = 1
    cmap[0,-1] = 0
    cmap[1:,-1] = 0.3

    
    img_out = cmap[pred.flatten()]
    R, C = pred.shape[:2]
    img_out = img_out.reshape((R, C, -1))

    img_out = Image.fromarray((img_out * 255).astype(np.uint8))
    img_org = Image.fromarray((img_org).astype(np.uint8)).convert('RGBA')    
    img_org.paste(img_out, (0,0), img_out)


    # predi = Image.fromarray((pred * 255).astype(np.uint8))

    ## SAVING IMAGE
    path = app.config['UPLOAD_FOLDER']
    name = Path(file).stem
    name = f'{name}_mask'                                   #example_mask
    img_org.save(f'{path}/preds/t2w_pred.png', 'PNG')
    # predi.save(f'{path}/preds/masks.png', 'PNG')

    np.save(os.path.join(app.config['UPLOAD_FOLDER'], name), pred)  #example_mask.npy
    pred = []
    return name


def save_png(name):
    """
    Saves input numpy array in nii.gz file.
    """
    dir = app.config['EXPORT_FOLDER']
    os.makedirs(dir, exist_ok=True)
    path = app.config['UPLOAD_FOLDER']
    Image.open(f'{path}/preds/t2w_pred.png')
    print(f"Succesfully saved {name}")
