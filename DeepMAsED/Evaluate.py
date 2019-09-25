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
    
    # 'final_model.h5' 
    h5_file = os.path.join(args.save_path, 'final_model.h5')
    if not os.path.exists(h5_file):
        msg = 'Cannot find {} file in {}'
        raise IOError(msg.format('final_model.h5', args.save_path))
    logging.info('Loading file: {}'.format(h5_file))
    model = load_model(h5_file, custom_objects=custom_obj)
    
    # prediction directory
    pred_dir = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # model pkl
    pkl_file = os.path.join(args.save_path, 'mean_std_final_model.pkl')
    logging.info('Loading file: {}'.format(pkl_file))
    with open(pkl_file, 'rb') as mstd:
        mean_tr, std_tr = pickle.load(mstd)
    
    # tech
    tech = args.technology
    auc = []

    # loading features
    if args.is_synthetic == 1:
        x, y, i2n = Utils.load_features(args.data_path,
                                        max_len = args.max_len,
                                        mode = args.mode, 
                                        technology = tech,
                                        force_overwrite=args.force_overwrite)
    else:
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
    
    logging.info('Computing predictions for {}...'.format(tech))    
    scores = compute_predictions(y, n2i, model, dataGen)
    outfile = os.path.join(args.save_path, 'predictions', tech + '.pkl')
    with open(outfile, 'wb') as spred:
        pickle.dump(scores, spred)
    logging.info('File written: {}'.format(outfile))

        
    # # old
    # for model_path in path_to_models:
    #     # finding final_model.h5 file
    #     h5_file = 'final_model.h5'
    #     D = os.path.join(args.save_path, model_path)
    #     F = os.path.join(D, h5_file)
    #     if not os.path.exists(F):
    #         msg = 'Cannot find {} file in {}'
    #         logging.warning(msg.format(h5_file, D))
    #         continue
    #     else:
    #         msg = 'Found {} file in {}'
    #         logging.info(msg.format(h5_file, D))

    #     # predictions dir
    #     if not os.path.exists(os.path.join(args.save_path, model_path, 'predictions')):
    #         os.makedirs(os.path.join(args.save_path, model_path, 'predictions'))
    
    #     if not os.path.exists(os.path.join(args.save_path, model_path, 
    #                                        'predictions', args.data_path.split('/')[-1])):
    #         os.makedirs(os.path.join(args.save_path, model_path, 
    #                                        'predictions', args.data_path.split('/')[-1]))

    #     F = os.path.join(args.save_path, model_path, 'mean_std_final_model.pkl')
    #     with open(F, 'rb') as mstd:
    #         mean_tr, std_tr = pickle.load(mstd)
    
    #     model = load_model(os.path.join(args.save_path, model_path, 'final_model.h5'), 
    #                        custom_objects=custom_obj)
    
    #     tech = args.technology
            
    #     logging.info('Loading data...')
    #     if args.is_synthetic == 1:
    #         x, y, i2n = Utils.load_features(args.data_path,
    #                                         max_len=args.max_len,
    #                                         mode = args.mode, 
    #                                         technology=tech)
    #     else:
    #         x, y, i2n = Utils.load_features_nogt(args.data_path,
    #                                              max_len=args.max_len,
    #                                              mode = args.mode)
    
    #     logging.info('Loaded {} contigs...'.format(len(set(i2n.values()))))
    
    #     n2i = Utils.reverse_dict(i2n)
    #     x = [xi for xmeta in x for xi in xmeta]
    #     y = np.concatenate(y)
    
    #     dataGen = Models.Generator(x, y, args.max_len, batch_size=64,  shuffle=False, 
    #                                norm_raw=bool(args.norm_raw),
    #                                mean_tr=mean_tr, std_tr=std_tr)
    
    #     loggin.info('Computing predictions for {}...'.format(tech))
    
    #     scores = compute_predictions(y, n2i)
    #     outfile = os.path.join(args.save_path, model_path, 'predictions', 
    #                            args.data_path.split('/')[-1],  tech + '.pkl')
    #     with open(outfile, 'wb') as spred:
    #         pickle.dump(scores, spred)
    #     logging.info('File written: {}'.format(outfile))
       
    
if __name__ == '__main__':
    pass
