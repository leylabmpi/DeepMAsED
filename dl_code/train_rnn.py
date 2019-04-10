import numpy as np
import keras
from sklearn.metrics import confusion_matrix

import argparse
import IPython

import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ani_path', default='data')
args = parser.parse_args()

max_len = 3000

data_path='/home/mrojas/deepmased/tests/output_n10/map/1/megahit'

# Build model
chi_net = models.chimera_net(max_len)

# Load and process data
x_tr, x_te, y_tr, y_te = utils.load_features(data_path, max_len=3000, 
                                             test_size=0.2)

#Train model
w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
class_weight = {0 : 1 , 1: w_one}

tb_logs = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                     write_graph=True, write_images=True)

chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=5, 
               class_weight=class_weight, callbacks=[tb_logs])

# Run predictions
pred_tr = (chi_net.net.predict(x_tr) > 0.5).astype(int)
pred_te = (chi_net.net.predict(x_te) > 0.5).astype(int)

print("Training")
print(confusion_matrix(pred_tr, y_tr) / len(pred_tr))
print(np.mean(pred_tr == y_tr))
print("Test")
print(confusion_matrix(pred_te, y_te) / len(pred_te))
print(np.mean(pred_te == y_te))

