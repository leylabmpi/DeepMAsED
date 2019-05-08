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
parser.add_argument('--filters', default=8, type=int, 
                    help='N of filters for first conv layer. Then x2.')
parser.add_argument('--n_conv', default=2, type=int, 
                    help='N of conv layers.')
parser.add_argument('--n_epochs', default=10, type=int, 
                    help='N of training epochs.')
parser.add_argument('--max_len', default=3000, type=int, 
                    help='Max contig len, fixed input for CNN.')
parser.add_argument('--dropout', default=0.1, type=float, 
                    help='Rate of dropout.')
parser.add_argument('--pool_window', default=40, type=int, 
                    help='Window size for average pooling.')
parser.add_argument('--test_size', default=0.3, type=float, 
                    help='Size of test set.')
parser.add_argument('--lr_init', default=0.001, type=float, 
                    help='Size of test set.')
parser.add_argument('--mode', default='chimera', type=str, 
                    help='Chimera or edit distance.')
args = parser.parse_args()

# Load and process data
print("Loading data...")
x_tr, x_te, y_tr, y_te = utils.load_features(args.data_path, max_len=args.max_len, 
                                             test_size=args.test_size, 
                                             mode=args.mode)
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}
model = load_model(os.path.join(args.save_path, 'model.h5'), 
                   custom_objects=custom_obj)

print("Saving ROC curves...")
score_tr = model.predict(x_tr)
score_te = model.predict(x_te)

f_tr, t_tr, th_tr  = roc_curve(y_tr, score_tr)
f_te, t_te, th_te  = roc_curve(y_te, score_te)

plt.plot(f_tr, t_tr)
plt.plot(f_te, t_te)

plt.savefig(os.path.join(args.save_path, 'roc_curve.pdf'))

IPython.embed()

