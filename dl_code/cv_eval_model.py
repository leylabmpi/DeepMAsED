import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse
import IPython

import models
import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

np.random.seed(12)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--save_path', default='model', type=str, 
                    help='Where to save training weights and logs.')
parser.add_argument('--save_plot', default=None, type=str, 
                    help='Where to save plots. Defaults to save_path if None.')
parser.add_argument('--max_len', default=10000, type=int, 
                    help='Max contig len, fixed input for CNN.')
parser.add_argument('--test_size', default=0.3, type=float, 
                    help='Size of test set.')
parser.add_argument('--mode', default='chimera', type=str, 
                    help='Chimera or edit distance.')
args = parser.parse_args()

save_plot = args.save_plot
if save_plot is None:
    save_plot = args.save_path

# Load and process data
print("Loading data...")
#x_tr, x_te, y_tr, y_te = utils.load_features(args.data_path, max_len=args.max_len, 
#                                             test_size=args.test_size, 
#                                             mode=args.mode)

x, y = utils.load_features(args.data_path,
                           test_size=args.test_size, 
                           max_len=args.max_len,
                            mode = args.mode)


# Provide objective to load
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}

path_to_models = os.listdir(args.save_path)
#path_to_models = ['nfilter_8_nconv_3_lr_0.0001_dropout_0.5_pool_20_nhid_20_nfc_3']

for model_path in path_to_models:
    print(model_path)
    val_cv = []

    for i in range(5):
        x_tr, x_val, y_tr, y_val = utils.kfold(x, y, 0)

        dataGen = models.Generator(x_val, y_val, args.max_len, batch_size=64,  
                                   shuffle=False)

        if not os.path.exists((os.path.join(args.save_path, 
                                            model_path, str(i) + '_model.h5'))):
            continue

        model = load_model(os.path.join(args.save_path, model_path, 
                           str(i) + '_model.h5'), 
                           custom_objects=custom_obj)

        score_val = model.predict_generator(dataGen)

        val_cv.append(roc_auc_score(y_val[0:score_val.size], score_val))
    print(np.mean(val_cv))
