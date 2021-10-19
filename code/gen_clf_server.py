import os
import pickle
from flask import Flask
from flask import request
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import json

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def pred_API(models_folder='./models/', tok_name='tok_1623170133_8569906',
             aux_folder='./aux/', id_genres_dict_file='id_genres_dict_1623170035_963645.pkl',
             model_name='model1_1623170131_5833135', conf_threshold=0.3):
    """
    wrapper for API POST inference
    """
    # load tokenizer
    tok_path = os.path.join(models_folder, tok_name)
    with open(tok_path, 'rb') as f:
        tok = pickle.load(f)

    # load genres dict
    with open(os.path.join(aux_folder, id_genres_dict_file), 'rb') as f:
        id_genres_dict = pickle.load(f)

    # load model
    model_path = os.path.join(models_folder, model_name)
    model = keras.models.load_model(model_path)
    # get inputs from req
    payload = request.json
    title = payload["title"]
    description = payload["description"]
    # preproc
    merged_desc = str(title)+" "+str(description)
    tok_inp = tok.texts_to_sequences([merged_desc])
    pad_inp = pad_sequences(tok_inp, maxlen=model.input_shape[1], padding='post', truncating='post')
    pred = np.array(model.predict(pad_inp)).flatten()
    pred_g = np.argwhere(pred>conf_threshold).flatten()
    if len(pred_g)==0: # if no genre with enough confidence, then softmax approachs
        pred_g = np.argmax(pred).flatten()
    genres = ""
    for g in pred_g:
        genres+="{}, ".format(id_genres_dict[g])
    genres = genres[:-2] # strip space and final comma
    # output dict -> json
    isgpu = tf.test.is_gpu_available()
    out_dict = {"is_gpu": isgpu, "title": title, "description": description, "genre": genres}
    return json.dumps(out_dict)

if __name__ == "__main__":
    app.run(host='0.0.0.0')