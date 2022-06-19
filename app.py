# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 00:49:03 2022

@author: Saqib
"""
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

import cv2 as cv
import matplotlib.pyplot as plt

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')
tf.compat.v1.enable_eager_execution()

import joblib
# local imports
import voxelmorph as vxm
import utils
from registration import RegistrationModel
from data_gen import DataGenerator
from feature_extraction import FeatureExtractor
from model import Model    
from joblib import dump, load

from flask import Flask

app = Flask(__name__)



@app.route("/")
def hello_world():
    return '<p>Hello, World!</p><img src="/test.png" alt="my plot"/>'

@app.route('/print-plot')
def plot_png():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   xs = np.random.rand(100)
   ys = np.random.rand(100)
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route("/test.png")
def test():
    
    model = RegistrationModel().model
    model.load_weights("./final-registr-left.h5")

    val_input, out = DataGenerator.vxm_data_generator("./")
    zero, val_idx = out
    
    
    # visualize
    val_pred = model.predict(val_input)
    
    images = [img[0, :, :, 0] for img in val_input + val_pred] 
    
    annotations_test = utils.get_annotations("./images/labels.csv")
     # Get outputs
    moving = images[0]
    fixed = images[1]
    moved = images[2]
    annotations = annotations_test[0]
    
    # get dense field of inverse affine
    field_inv = val_pred[1].squeeze()
    field_inv = field_inv[np.newaxis, ...]
    annotations_keras = annotations[np.newaxis, ...]
    
    # warp annotations
    data = [tf.convert_to_tensor(f, dtype=tf.float32) for f in [annotations_keras, field_inv]]
    annotations_warped = vxm.utils.point_spatial_transformer(data)[0, ...].numpy()
    
    # New Prediciton
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(fixed, cmap='gray')
    plt.plot(*[annotations[:, f] for f in [0, 1]], 'o',color="red")  
    plt.title("Template Image \n (with annotations)")
    
    plt.subplot(1,2,2)
    plt.imshow(moving, cmap='gray')
    plt.plot(*[annotations_warped[:, f] for f in [0, 1]], 'o', color="red")
    plt.title("New Image \n (with predicted annotations)")
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
    utils.draw_patches(moving, annotations_warped)
    fe = FeatureExtractor()
    feat_df = fe.generate_features(moving, annotations_warped)
    
    m = Model()
    
    clf = load('xgb-LEFT-140.0-FEATURES-AUG1-clf.joblib') 
    feat_names = clf.get_booster().feature_names
    scaled_values = m.get_scaler().transform(feat_df.iloc[0].values.reshape(1, -1))
    feat_df.loc[0] = scaled_values
   
    m.generate_prediction(feat_df[feat_names])



if __name__ == "__main__":
    app.run(debug=False)
    
    