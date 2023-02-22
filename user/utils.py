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
from tensorflow.keras.layers import Dropout
import cv2
import os
from multiprocessing.pool import ThreadPool
from config import Config
import segmentation_models as sm
from PIL import Image
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from static.models.mcdropout import MCDropout
K.set_image_data_format('channels_first')

path = app.config['UPLOAD_FOLDER']
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


def compute_entropy(predictive_prob):
    entropy_func = lambda x: -1 * np.sum(np.log(x + np.finfo(np.float32).eps) * x, axis=3)
    return entropy_func(predictive_prob)

def uncertainty(img):
    global model
    T = app.config["T"]
    N_class = app.config["NO_ZONES"]
    predictive_prob_total = np.zeros((1, 256, 256, N_class))
    for i in range(T):
        predictive_prob = model.predict(img, verbose=0)
        if (type(predictive_prob) is list):# some models may return logit, segmap
            predictive_prob = predictive_prob[1]
        predictive_prob_total += predictive_prob

    pred_prob_avg = predictive_prob_total / (T * 1.0)
    entropy = compute_entropy(pred_prob_avg)[0,:,:]
    pred_argmax = np.argmax(pred_prob_avg, axis=3)[0,:,:]
    return pred_argmax, entropy


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'iou_score': metrics[0],
                      'f1-score': metrics[1],
                      "MCDropout": MCDropout(Dropout)}
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

def get_entropy_img(entropy_data):
    entropy = cm.jet(entropy_data)
    entropy = Image.fromarray((entropy * 255).astype(np.uint8)).convert('RGBA')
    data_uq = entropy.getdata()
    newData_uq = []
    for item in data_uq:
        if (item[0] == 0 and item[1] == 0 and item[2] == 127 and item[3] == 255):
            newData_uq.append((255, 255, 255, 0))
        else:
            newData_uq.append(item)
    entropy.putdata(newData_uq)
    return entropy

def postprocess_pred(prediction):
    zones = app.config['NO_ZONES']
    colours = cm.get_cmap('hsv', zones)
    cmap = colours(np.linspace(0,1,zones))
    cmap[0,:] = 1
    cmap[0,-1] = 0
    cmap[1:,-1] = 0.3

    img_pred = cmap[prediction.flatten()]
    R, C = prediction.shape[:2]
    img_pred = img_pred.reshape((R, C, -1))
    return img_pred

def colorbar_img(entropy_data, uq_img):
    # path = app.config['UPLOAD_FOLDER']
    global path
    mpb = plt.pcolormesh(entropy_data, cmap='jet')
    fig,ax = plt.subplots(figsize=(4,3))
    plt.colorbar(mpb,ax=ax, orientation='horizontal')
    ax.remove()
    plt.savefig(f'{path}/preds/onlycbar.png',bbox_inches='tight')
    colorbar = Image.open(f'{path}/preds/onlycbar.png')
    cb_size = colorbar.size
    uq_size = uq_img.size
    colorbar = colorbar.resize((uq_size[0], cb_size[1]))
    size = (uq_size[0], uq_size[1]+cb_size[1])
    uq_full = Image.new("RGB", size, "white")
    uq_full.paste(uq_img, (0, 0))
    uq_full.paste(colorbar, (0, uq_size[1]))
    uq_full.save(f'{path}/preds/uncertainty.png', 'PNG', dpi=(300,300))
    [os.remove(os.path.join(path,'preds/onlycbar.png'))]


def make_png(file):
    global model, path
    pred = []
    if 'model' not in globals():
        model = load_old_model(app.config['MODEL'])

    img_org = np.load(f'{path}/npy/{file}')
    img = preprocess(img_org)
    # pred = predict(img)                       # Used for regular models (without montecarlo dropout)
    
    pred, entropy = uncertainty(img)            # Used for MCDropout models
    
    ## PLOT IMAGE AND PRED
    img_out = postprocess_pred(pred)

    # PREDICTION IMAGE SAVE
    img_out = Image.fromarray((img_out * 255).astype(np.uint8))
    img_org_1 = Image.fromarray((img_org).astype(np.uint8)).convert('RGBA')    
    img_org_1.paste(img_out, (0,0), img_out)

    # ENTROPY IMAGE SAVE
    img_org_2 = Image.fromarray((img_org).astype(np.uint8)).convert('RGBA')  
    entropy_img = get_entropy_img(entropy)
    img_org_2.paste(entropy_img, (0,0), entropy_img)

    ## Colorbar image save
    colorbar_img(entropy, img_org_2)

    ## SAVING IMAGE
    name = Path(file).stem
    name = f'{name}_mask'                                           #example_mask
    img_org_1.save(f'{path}/preds/t2w_pred.png', 'PNG')

    # np.save(os.path.join(path, name), pred)  #example_mask.npy
    pred = []
    return name


def save_png(name):
    """
    Saves input numpy array in nii.gz file.
    """
    global path
    dir = app.config['EXPORT_FOLDER']
    os.makedirs(dir, exist_ok=True)
    # path = app.config['UPLOAD_FOLDER']
    Image.open(f'{path}/preds/t2w_pred.png')
    print(f"Succesfully saved {name}")
