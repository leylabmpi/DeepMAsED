import numpy as np
import keras
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score

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

class Config(object):
    max_len = args.max_len
    filters = args.filters
    n_conv = args.n_conv
    n_features = 9
    pool_window = args.pool_window
    dropout = args.dropout
    lr_init = args.lr_init
    mode = args.mode

# Build model
config = Config()

chi_net = models.chimera_net(config)
chi_net.print_summary()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

save_path = args.save_path

# Load and process data
x_tr, x_te, y_tr, y_te = utils.load_features(args.data_path, max_len=args.max_len, 
                                             test_size=args.test_size, 
                                             mode = config.mode)
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
y_tr = st.fit_transform(y_tr)
y_te = st.transform(y_te)

#Train model
tb_logs = keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'), 
                                     histogram_freq=0, 
                                     write_graph=True, write_images=True)

if config.mode == 'chimera':
    w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
    class_weight = {0 : 1 , 1: w_one}


    chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=args.n_epochs, 
                   class_weight=class_weight, 
                   callbacks=[tb_logs, chi_net.reduce_lr])

elif config.mode == 'edit':
    chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=args.n_epochs, 
                   callbacks=[tb_logs, chi_net.reduce_lr])

IPython.embed()
chi_net.save(os.path.join(save_path, 'model.h5'))
"""
# Run predictions
scores_tr = chi_net.predict(x_tr)
scores_te = chi_net.predict(x_te)

pred_tr = (scores_tr > 0.5).astype(int)
pred_te = (scores_te > 0.5).astype(int)

print("Training: confusion matrix and accuracy")
c_tr = confusion_matrix(y_tr, pred_tr)
norm_c_tr = (c_tr.T / np.sum(c_tr, axis=1)).T
print(norm_c_tr)
print("%.2f" % np.mean(pred_tr == y_tr))


print()
print("Test: confusion matrix and accuracy")
c_te = confusion_matrix(y_te, pred_te)
norm_c_te = (c_te.T / np.sum(c_te, axis=1)).T
print(norm_c_te)
print("%.2f" % np.mean(pred_te == y_te))
"""

