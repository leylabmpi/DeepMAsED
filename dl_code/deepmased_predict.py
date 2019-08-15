import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse
import IPython
import _pickle as pickle

import models
import utils

import os

np.random.seed(12)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--save_path', default='model', type=str, 
                    help='Where to save training weights and logs.')
args = parser.parse_args()


# Load and process data
# Provide objective to load
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}

model_path = args.save_path

auc = []

if not os.path.exists((os.path.join(args.save_path, model_path, 'deepmased.h5'))):
    print("Model file not available")
    exit()

if not os.path.exists(os.path.join(args.save_path, model_path, 
                                   'predictions')):
    os.makedirs(os.path.join(args.save_path, model_path, 'predictions'))

if not os.path.exists(os.path.join(args.save_path, model_path, 
                                   'predictions', args.data_path.split('/')[-1])):
    os.makedirs(os.path.join(args.save_path, model_path, 
                                   'predictions', args.data_path.split('/')[-1]))

with open(os.path.join(args.save_path, model_path, 'mean_std_final_model.pkl'), 'rb') as mstd:
    mean_tr, std_tr = pickle.load(mstd)


model = load_model(os.path.join(args.save_path, model_path, 'deepmased.h5'), 
                   custom_objects=custom_obj)

print("Loading data...")
x, y, i2n = utils.load_features_nogt(args.data_path)

print("Loaded %d contigs..." % len(set(i2n.values())))

n2i = utils.reverse_dict(i2n)
x = [xi for xmeta in x for xi in xmeta]
y = np.concatenate(y)

dataGen = models.Generator(x, y, batch_size=64,  shuffle=False, 
                           norm_raw=0,
                           mean_tr=mean_tr, std_tr=std_tr)

print("Computing predictions...")
save_path = os.path.join(args.save_path, model_path, 'predictions', args.data_path.split('/')[-1])
scores = utils.compute_predictions(n2i, dataGen, model, save_path)


