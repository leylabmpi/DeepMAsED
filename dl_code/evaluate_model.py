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
parser.add_argument('--mode', default='chimera', type=str, 
                    help='Chimera or edit distance.')
parser.add_argument('--technology', default='megahit', type=str, 
                    help='Megahit or Metaspades.')
parser.add_argument('--norm_raw', default=1, type=int, 
                    help='Whether to normalize the four one-hot feature of raw.')
args = parser.parse_args()

save_plot = args.save_plot
if save_plot is None:
    save_plot = args.save_path

# Load and process data
#dataGen_tr = models.Generator(x_tr, y_tr, args.max_len, batch_size=32,  shuffle=False)
# Provide objective to load
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}

print("Saving ROC curves...")
path_to_models = os.listdir(args.save_path)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
auc = []
for model_path in path_to_models:
    if not os.path.exists((os.path.join(args.save_path, model_path, 'final_model.h5'))):
        continue
    model = load_model(os.path.join(args.save_path, model_path, 'final_model.h5'), 
                       custom_objects=custom_obj)

    for tech in ['megahit', 'metaspades']:
        
        print("Loading data...")

        x, y = utils.load_features('../tests/train_n1000_r30',
                                   max_len=args.max_len,
                                    mode = args.mode, 
                                    technology=tech)

        x = [xi for xmeta in x for xi in xmeta]
        y = np.concatenate(y)
        dataGen = models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
                                   norm_raw=bool(args.norm_raw))

        mean_tr = dataGen.mean
        std_tr = dataGen.std

        x, y = utils.load_features(args.data_path,
                                   max_len=args.max_len,
                                    mode = args.mode, 
                                    technology=tech)

        x = [xi for xmeta in x for xi in xmeta]
        y = np.concatenate(y)

        dataGen = models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
                                   norm_raw=bool(args.norm_raw),
                                   mean_tr=mean_tr, std_tr=std_tr)
        #dataGen.mean = mean_tr
        #dataGen.std = std_tr

        
        print("Computing predictions...")
        score_val = model.predict_generator(dataGen, steps=500)
        auc.append(round(roc_auc_score(y[0:score_val.size], score_val), 2))
        print(auc[-1])
        exit()

        #f_tr, t_tr, th_tr  = roc_curve(y_tr[0 : score_tr.size], score_tr)
        f_val, t_val, th_val  = roc_curve(y[0 : score_val.size], score_val)

        #ax.plot(f_tr, t_tr, 'o--')
        ax.plot(f_val, t_val, 'o--', ms=2)

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

    plt.legend([r'MEGAHIT: AUC=' + str(auc[0]), r'MetaSPAdes: AUC=' + str(auc[1])], fontsize=18, 
                loc='lower right')

    if not os.path.exists(save_plot):
        os.makedirs(save_plot)

    plt.savefig(os.path.join(save_plot, 'roc_curve' + model_path + '.pdf'), 
                bbox_inches='tight', format='pdf', dpi=5000)

