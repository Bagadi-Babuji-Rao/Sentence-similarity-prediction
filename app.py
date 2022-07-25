# -*- coding: utf-8 -*-
import requests
# %tb
from flask import Flask, render_template, request
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        x = str(request.form['x'])
        y =str(request.form['y'])
        x_en =model.encode(x)
        y_en = model.encode(y)
        
        pred = cosine_similarity(np.array(x_en).reshape(1,-1),np.array(y_en).reshape(1,-1))[0][0]
        return render_template('index.html',prediction_text="Similarity Score:{}".format(pred))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True,use_reloader=False)