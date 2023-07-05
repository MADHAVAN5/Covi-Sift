import os
import re

import numpy as np
import requests
from flask import (Flask, flash, make_response, redirect, render_template,
                   request, session, url_for)
from tensorflow.keras import models
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat

from app import *

model = load_model(r'Model/InceptionV3-covid.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/upload', methods=['GET'])
def UploadGet():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def UploadPost():
    print('!!')

    file = request.files['file']
    if file.filename == '' :
        flash('No image selected for uploading')
        return redirect(request.url)   
    
    else:
        file.save(file.filename)
        img = image.load_img(file.filename,target_size = (299,299))
        x = image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        precaution = np.argmax(model.predict(img_data),axis=1)
        
        label = ['COVID','Lung_Capacity','Normal','Viral Pneumoniya']
        result = str(label[precaution[0]])
    return render_template('upload.html',result=result)