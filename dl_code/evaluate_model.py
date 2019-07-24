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


def compute_predictions(y, n2i):
    """
    Computes predictions for a model and generator, aggregating scores for long contigs.

    Inputs: 
        n2i: dictionary with contig_name -> list of idx corresponding to that contig.
    Output:
        score_agg: scores for individual contigs
        y_agg: corresponding true labels
    """

    score_val = model.predict_generator(dataGen)

    # Compute predictions by aggregating scores for longer contigs
    score_val = score_val.flatten()
    scores = {}
    for k in n2i:
        inf = n2i[k][0]
        sup = n2i[k][-1] + 1
        if k[0] not in scores:
            scores[k[0]] = {}
       
        # Make sure contig doesnt appear more than once
        assert(k[1] not in scores[k[0]])

        # Make sure we have predictions for these indices
        if sup > len(score_val):
            continue

        # Make sure all the labels for the contig coincide
        assert((y[inf : sup] == y[inf]).all())
        scores[k[0]][k[1]] = {'y' : int(y[inf]), 'pred' : score_val[inf : sup]}

    return scores


save_plot = args.save_plot
if save_plot is None:
    save_plot = args.save_path


# Load and process data
#dataGen_tr = models.Generator(x_tr, y_tr, args.max_len, batch_size=32,  shuffle=False)
# Provide objective to load
recall_0 = utils.class_recall(0)
recall_1 = utils.class_recall(1)
custom_obj = {'metr' : recall_0}

path_to_models = os.listdir(args.save_path)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
auc = []
for model_path in path_to_models:
    if not os.path.exists((os.path.join(args.save_path, model_path, 'final_model.h5'))):
        continue

    if not os.path.exists(os.path.join(args.save_path, model_path, 
                                       'predictions')):
        os.makedirs(os.path.join(args.save_path, model_path, 'predictions'))

    if not os.path.exists(os.path.join(args.save_path, model_path, 
                                       'predictions', args.data_path.split('/')[-1])):
        os.makedirs(os.path.join(args.save_path, model_path, 
                                       'predictions', args.data_path.split('/')[-1]))

    with open(os.path.join(args.save_path, model_path, 'mean_std_final_model.pkl'), 'rb') as mstd:
        mean_tr, std_tr = pickle.load(mstd)


    model = load_model(os.path.join(args.save_path, model_path, 'final_model.h5'), 
                       custom_objects=custom_obj)

    tech = args.technology
        
    print("Loading data...")

    x, y, i2n = utils.load_features(args.data_path,
                               max_len=args.max_len,
                                mode = args.mode, 
                                technology=tech)
    print("Loaded %d contigs..." % len(set(i2n.values())))

    n2i = utils.reverse_dict(i2n)
    x = [xi for xmeta in x for xi in xmeta]
    y = np.concatenate(y)

    dataGen = models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
                               norm_raw=bool(args.norm_raw),
                               mean_tr=mean_tr, std_tr=std_tr)
    #dataGen.mean = mean_tr
    #dataGen.std = std_tr

    
    print("Computing predictions for " + tech + " ...")

    scores = compute_predictions(y, n2i)

    with open(os.path.join(args.save_path, model_path, 'predictions', 
                           args.data_path.split('/')[-1],  tech + '.pkl'), 'wb') as spred:
        pickle.dump(scores, spred) 
    
