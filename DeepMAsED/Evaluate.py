# import
## batteries
import os
import sys
import logging
import _pickle as pickle
#import argparse
## 3rd party
import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import IPython
## application
from DeepMAsED import Models
from DeepMAsED import Utils


def compute_predictions(y, n2i, model, dataGen):
    """
    Computes predictions for a model and generator, NOT aggregating scores for long contigs.

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

def main(args):
    """Main interface
    """
    np.random.seed(12)
    logging.info('Loading data...')
    
    # where to save the plot
    save_plot = args.save_plot
    if save_plot is None:
        save_plot = args.save_path        
        
    # Load and process data
    # Provide objective to load
    recall_0 = Utils.class_recall(0)
    recall_1 = Utils.class_recall(1)
    custom_obj = {'metr' : recall_0}
    
    h5_file = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(h5_file):
        msg = 'Cannot find {} file in {}'
        raise IOError(msg.format(args.model_name, args.model_path))
    logging.info('Loading model: {}'.format(h5_file))
    model = load_model(h5_file, custom_objects=custom_obj)

    
    # model pkl
    pkl_file = os.path.join(args.model_path, args.mstd_name)
    logging.info('Loading file: {}'.format(pkl_file))
    with open(pkl_file, 'rb') as mstd:
        mean_tr, std_tr = pickle.load(mstd)
    

    # loading features
    if args.is_synthetic == 1:
        logging.info('Loading synthetic features')
        x, y, i2n = Utils.load_features(args.data_path,
                                        max_len = args.max_len,
                                        mode = args.mode, 
                                        technology = args.technology,
                                        force_overwrite=args.force_overwrite)
    else:
        logging.info('Loading non-synthetic features')
        x, y, i2n = Utils.load_features_nogt(args.data_path,
                                             max_len = args.max_len,
                                             mode = args.mode,
                                             force_overwrite=args.force_overwrite)
        
    logging.info('Loaded {} contigs...'.format(len(set(i2n.values()))))    
    n2i = Utils.reverse_dict(i2n)
    x = [xi for xmeta in x for xi in xmeta]
    y = np.concatenate(y)    
    dataGen = Models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
                               norm_raw=bool(args.norm_raw),
                               mean_tr=mean_tr, std_tr=std_tr)
    
    logging.info('Computing predictions for {}...'.format(args.technology))    
    scores = compute_predictions(y, n2i, model, dataGen)
    outfile = os.path.join(args.save_path, '_'.join([args.save_name, args.technology + '.pkl']))
    with open(outfile, 'wb') as spred:
        pickle.dump(scores, spred)
    logging.info('File written: {}'.format(outfile))

    
if __name__ == '__main__':
    pass
