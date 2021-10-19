import os
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse

def run_pred(model, tok, title, description, conf_threshold, id_genres_dict):
    """
    asks for input title and description and
    performs inference on movies genres with model 'model'
    and tokenizer 'tok'
    conf_threshold: float in [0, 1] defining threshold for genre confidence
    """
    merged_desc = str(title)+" "+str(description)
    tok_inp = tok.texts_to_sequences([merged_desc])
    pad_inp = pad_sequences(tok_inp, maxlen=model.input_shape[1], padding='post', truncating='post')
    pred = np.array(model.predict(pad_inp)).flatten()
    pred_g = np.argwhere(pred>conf_threshold).flatten()
    if len(pred_g)==0: # if no genre with enough confidence, then softmax approachs
        pred_g = np.argmax(pred).flatten()
    print()
    genres = ""
    for g in pred_g:
        genres+="{}, ".format(id_genres_dict[g])
    genres = genres[:-2] # strip space and final comma
    out_dict = {"title": title, "description": description, "genre": genres}
    return out_dict


def main(title, description, models_folder, tok_name, aux_folder, id_genres_dict_file, model_name, conf_threshold):
    """
    main function calling inference aux function run_pred
    """

    # load tokenizer
    tok_path = os.path.join(models_folder, tok_name)
    with open(tok_path, 'rb') as f:
        loaded_tok = pickle.load(f)

    # load genres dict
    with open(os.path.join(aux_folder, id_genres_dict_file), 'rb') as f:
        id_genres_dict = pickle.load(f)

    # load model
    model_path = os.path.join(models_folder, model_name)
    loaded_model = keras.models.load_model(model_path)

    out = run_pred(loaded_model, loaded_tok, title, description, conf_threshold, id_genres_dict)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', '-t', help='movie title')
    parser.add_argument('--description', '-d', help='movie description')
    parser.add_argument('--modfol', '-mf', help='folder where models are stored', default = './models/')
    parser.add_argument('--tokname', '-tn', help='ftokenizer file name', default = 'tok_1623170133_8569906')
    parser.add_argument('--auxfolder', '-af', help='aux folder path', default = './aux/')
    parser.add_argument('--idgd', '-idgd', help='id_genres_dict file name to be found in aux', default = 'id_genres_dict_1623170035_963645.pkl')
    parser.add_argument('--modelname', '-mn', help='tf model file name in models folder', default = 'model1_1623170131_5833135')
    parser.add_argument('--confthresh', '-ct', help='confidence threshold (float in [0, 1])', default = 0.3)
    args = parser.parse_args()
    title = str(args.title)
    description = str(args.description)
    models_folder = str(args.modfol)
    tok_name = str(args.tokname)
    aux_folder = str(args.auxfolder)
    id_genres_dict_file = str(args.idgd)
    model_name = str(args.modelname)
    conf_threshold = float(args.confthresh)
    main(title, description, models_folder, tok_name, aux_folder, id_genres_dict_file, model_name, conf_threshold)



