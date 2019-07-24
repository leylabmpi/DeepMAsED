import numpy as np
from scipy.misc import logsumexp
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse
import IPython
import csv

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_shuffle_seq_ref_function
#from deeplift.util import randomly_shuffle_seq
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle

import models
import utils

import pyBigWig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

from deeplift.visualization import viz_sequence
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

#Create directory for deeplift scores
bw_dir_path = os.path.join(args.save_path, 'deeplift')
if not os.path.exists(bw_dir_path):
    os.makedirs(bw_dir_path)
bw_dir_path = os.path.join(bw_dir_path, args.data_path.split('/')[-1])
if not os.path.exists(bw_dir_path):
    os.makedirs(bw_dir_path)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def length_from_scores(x, scores, thr, is_max):
    sort_args_score = np.argsort(scores)[::-1]
    cont_len = []
    for i in range(thr):
        if is_max == 1:
            idx = sort_args_score[i]
        else:
            idx = sort_args_score[-i -1]

        cap = x[idx].shape[0]
        test_x = x[idx]
        cont_len.append(np.amax(test_x[:, -2]))
    return cont_len

def get_lift_scores(idx, x, mean, std, num_refs=10, batch_size=64):
    """
    Compute scores from DeepLIFT for example at location idx. 
    """
    cap = x[idx].shape[0]
    test_x = x[idx]
    test_x = (test_x - mean) / std
    test_x = np.concatenate([test_x, np.zeros((10000 - x[idx].shape[0], 11))], 0)
    test_x = test_x[None, :, :, None]

    scores = rescale_conv_revealcancel_fc_many_refs_func(task_idx=0, 
      input_data_sequences=test_x, 
      num_refs_per_seq=num_refs,
      batch_size=batch_size)

    return scores, test_x, cap

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
    model = load_model(os.path.join(args.save_path, model_path, 'final_model.h5'), 
                       custom_objects=custom_obj)
    print(model.summary())


    for tech in ['megahit', 'metaspades']:
        
        print("Loading data...")

        x, y, i2n = utils.load_features(args.data_path,
                                        max_len=args.max_len,
                                        mode = args.mode, 
                                        technology=tech)
        
        x = [xi for xmeta in x for xi in xmeta]
        y = np.concatenate(y)

        n2i = utils.reverse_dict(i2n)

        print("Num contigs: %d \n" % len(x))

        #Compute predictions
        dataGen = models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
                                   norm_raw=bool(args.norm_raw))
        mean = dataGen.mean
        std = dataGen.std

        score_dm = model.predict_generator(dataGen).flatten()
      
        #Lenght of contigs
        cont_len_max = length_from_scores(x, score_dm, 10, 1)
        print("Max coverage of contigs with HIGHER confidence of misassembly")
        print(cont_len_max)
        

        print("\nMax of contigs with LOWER confidence of misassembly")
        cont_len_min = length_from_scores(x, score_dm, 10, -1)
        print(cont_len_min)

        # Predict using max coverage
        pred_naive = []
        thr = 7
        for i in range(len(x)):
            pred_naive.append(np.amax(x[i][:, -2]))
  
        pred_naive = np.array(pred_naive)
        pred_naive = pred_naive / sum(pred_naive)
        score_naive = sigmoid(pred_naive)
        auc_naive = round(roc_auc_score(y, score_naive), 2)
        print("\nAUC of naive predictor using max contig length as a score: %f" % auc_naive)
        auc_dm = round(roc_auc_score(y[0 : score_dm.size], score_dm), 2)
        print("\nAUC of DeepMASED: %f" % auc_dm)

        #f_val, t_val, th_val  = roc_curve(y[0 : score_dm.size], score_dm)

        # Build DeepLIFT model
        deeplift_model = kc.convert_model_from_saved_files(
          os.path.join(args.save_path, model_path, 'final_model.h5'), 
          nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 

        find_scores_layer_idx = 0
        deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-3)

        rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
          score_computation_function=deeplift_contribs_func,
          shuffle_func=dinuc_shuffle)

        # Rank predictions from DM
        sort_args_score = np.argsort(score_dm)[::-1]
        mean_ac = []
        ave_one = np.zeros(11)
       
        num_printed = 10
        plt_max_min = 3
        
        headers = []
        scores_lift = {}

        seen_idxs = set([])
        print("\nPosition with high scores on higher confidence of misassembly")
        for j in range(num_printed):
            idx_one = sort_args_score[j]
            if idx_one in seen_idxs:
                continue
        
            idx_all = n2i[i2n[idx_one]]
            scores_lift_high = []

            # Compute predictions for whole contig

            for idx in idx_all:
                # Add to seen
                seen_idxs.add(idx)
                scores_lift_high_i, test_x, cap = get_lift_scores(idx, x, mean, std)
                scores_lift_high_i = np.sum(scores_lift_high_i[0, 0:cap, :, 0], 1)
                scores_lift_high.append(scores_lift_high_i)

            assembly = str(i2n[idx_one][0])
            if assembly not in scores_lift:
                scores_lift[assembly] = {}

            scores_lift_high = np.concatenate(scores_lift_high)
            #scores_lift.append(scores_lift_high)

            #headers.append((str(i2n[idx_one][0]), i2n[idx_one][1], len(scores_lift_high)))
            contig = i2n[idx_one][1]
            scores_lift[assembly][contig] = scores_lift_high

            sorted_scores = np.argsort(scores_lift_high)[::-1]

            """
            print("\nTrue label: %d" % y[idx_one])
            for num in range(2):
                print(np.around(test_x[0, sorted_scores[num], :, 0] * std + mean).astype(np.uint8))
            for num in range(2):
                print(np.around(test_x[0, sorted_scores[-num-1], :, 0] * std + mean).astype(np.uint8))
            """

            """
            fig, ax = plt.subplots(figsize=(10, 10))
            #im = ax.imshow(scores_lift_high[:, None], cmap='bwr', aspect='auto', norm=matplotlib.colors.Normalize(vmin=-plt_max_min, vmax=plt_max_min))
            ax.plot(scores_lift_high)
            #fig.colorbar(im, ax=ax)
            #cbar = fig.colorbar(im, extend='both')
            #cbar.cmap.set_over('green')
            plt.savefig('plots/interp_pos_'+str(j) +'.pdf')
            with open('tmp_mod/csv_try.csv', 'w') as csv_file:
                fields = ['Metagenome', 'contig_name', 'deeplift_score']
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                for u in range(len(scores_lift_high)):
                    writer.writerow({fields[0] : i2n[idx_one][0], 
                                     fields[1] : i2n[idx_one][1], 
                                     fields[2] : scores_lift_high[u]})
            """
        
        # Save files
        for k in scores_lift:
            if not os.path.exists(os.path.join(bw_dir_path, k)):
                os.makedirs(os.path.join(bw_dir_path, k))

            bw = pyBigWig.open(os.path.join(bw_dir_path, k, tech + '.bw'), 'w')

            #Create headers
            headers = []
            for cont in scores_lift[k]:
                headers.append((cont, len(scores_lift[k][cont])))

            bw.addHeader(headers)
            for cont in scores_lift[k]:
                bw.addEntries(cont, np.arange(len(scores_lift[k][cont])), 
                              values=scores_lift[k][cont], span=1)
        continue
        exit()


        print("\nPosition with high scores on lower confidence of misassembly")
        for j in range(um_printed):
            idx_zero = sort_args_score[-j - 1]
            scores_lift_low, test_x, cap = get_lift_scores(idx_zero, x, mean, std)
            scores_lift_low = np.sum(scores_lift_low[0, 0:cap, :, 0], 1)

            sorted_scores = np.argsort(scores_lift_low)[::-1]
            print("\nTrue label: %d" % y[idx_zero])
            for num in range(2):
                print(np.around(test_x[0, sorted_scores[num], :, 0] * std + mean).astype(np.uint8))
            for num in range(2):
                print(np.around(test_x[0, sorted_scores[-num-1], :, 0] * std + mean).astype(np.uint8))

            fig, ax = plt.subplots(figsize=(10, 10))
            #im = ax.imshow(scores_lift_low[:, None], cmap='bwr', aspect='auto', norm=matplotlib.colors.Normalize(vmin=-plt_max_min, vmax=plt_max_min))

            ax.plot(scores_lift_high)
            #fig.colorbar(im, ax=ax)
            #cbar = fig.colorbar(im, extend='both')
            #cbar.cmap.set_over('green')
            plt.savefig('plots/interp_neg_'+str(j) +'.pdf')
        print(np.round(ave_one / float(10), 2))
        exit()
