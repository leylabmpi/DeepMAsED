import numpy as np
import keras
from sklearn.metrics import confusion_matrix, roc_curve

import argparse
import IPython

import models
import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(12)

parser = argparse.ArgumentParser()
parser.add_argument('--ani_path', default='data')
args = parser.parse_args()


data_path='/home/mrojas/deepmased/tests/output_n10/map/1/megahit'

class Config(object):
    max_len = 3000
    test_size = 0.3
    filters = [8, 16, 32]
    n_features = 9
    pool_window = 40
    dropout = 0.1

# Build model
config = Config()

chi_net = models.chimera_net(config)
chi_net.print_summary()

# Load and process data
x_tr, x_te, y_tr, y_te = utils.load_features(data_path, max_len=3000, 
                                             test_size=0.4)

#Train model
w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
class_weight = {0 : 1 , 1: w_one}

tb_logs = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                     write_graph=True, write_images=True)

chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=12, 
               class_weight=class_weight, callbacks=[tb_logs])

# Run predictions
scores_tr = chi_net.predict(x_tr)
scores_te = chi_net.predict(x_te)

pred_tr = (scores_tr > 0.5).astype(int)
pred_te = (scores_te > 0.5).astype(int)

print("Training")
c_tr = confusion_matrix(y_tr, pred_tr)
norm_c_tr = (c_tr.T / np.sum(c_tr, axis=1)).T
print(norm_c_tr)
print(np.mean(pred_tr == y_tr))
print("Test")
c_te = confusion_matrix(y_te, pred_te)
norm_c_te = (c_te.T / np.sum(c_te, axis=1)).T
print(norm_c_te)
print(np.mean(pred_te == y_te))

#ROC curve
fpr, tpr, th = roc_curve(y_tr, scores_tr, pos_label=1)
plt.plot(fpr, tpr)
plt.savefig('/is/cluster/mrojas/tmp/roc_train.pdf')

fpr, tpr, th = roc_curve(y_te, scores_te, pos_label=1)
plt.plot(fpr, tpr)
plt.savefig('/is/cluster/mrojas/tmp/roc_test.pdf')


