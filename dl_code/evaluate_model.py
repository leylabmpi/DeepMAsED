import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score
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
parser.add_argument('--max_len', default=3000, type=int, 
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
x_tr, x_te, y_tr, y_te = utils.load_features(args.data_path, max_len=args.max_len, 
                                             test_size=args.test_size, 
                                             mode=args.mode)
# Provide objective to load
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}

print("Saving ROC curves...")
path_to_models = os.listdir(args.save_path)
#path_to_models = ['nfilter_8_nconv_3_lr_0.0001_dropout_0.5_pool_40_nhid_40_nfc_3']

for model_path in path_to_models:
    if not os.path.exists((os.path.join(args.save_path, model_path, 'model.h5'))):
        continue
    model = load_model(os.path.join(args.save_path, model_path, 'model.h5'), 
                       custom_objects=custom_obj)

    score_tr = model.predict(x_tr)
    score_te = model.predict(x_te)

    f_tr, t_tr, th_tr  = roc_curve(y_tr, score_tr)
    f_te, t_te, th_te  = roc_curve(y_te, score_te)

    """
    for th in [0.1, 0.2, 0.25, 0.3, 0.6]:
        idx = np.where(f_te >= th)[0][0]
        pred_te = (score_te > th_te[idx]).astype(int)
        print("Confusion matrix with %f false positives" % th)
        print(confusion_matrix(y_te, pred_te))
        print(t_te[idx - 1])
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(f_tr, t_tr, 'o--')
    ax.plot(f_te, t_te, 'o--')

    # Set ticks
    ticks = np.arange(0, 1.1, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticks = [r"$" + str(t) + "$" for t in ticks]
    ax.set_xticklabels(ticks, fontsize=22)
    ax.set_yticklabels(ticks, fontsize=22)
    ax.set_xlabel(r'False positives', fontsize=22)
    ax.set_ylabel(r'True positives', fontsize=22)

    ax.grid(True, axis='y', linestyle=':')
    ax.grid(True, axis='x', linestyle=':')

    plt.legend([r'Training data', r'Test data'], fontsize=18, 
                loc='lower right')

    if not os.path.exists(save_plot):
        os.makedirs(save_plot)

    plt.savefig(os.path.join(save_plot, 'roc_curve' + model_path + '.pdf'), 
                bbox_inches='tight', format='pdf', dpi=5000)

