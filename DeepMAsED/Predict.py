# import
## batteries
import os
import sys
import logging
## 3rd party
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import IPython
import _pickle as pickle
## application
from DeepMAsED import Models
from DeepMAsED import Utils


def main(args):
    np.random.seed(args.seed)

    # CPU only instead of GPU
    if args.cpu_only:
        logging.info('Setting env for CPU-only mode...')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    # Load and process data
    # Provide objective to load
    recall_0 = Utils.class_recall(0)
    recall_1 = Utils.class_recall(1)
    custom_obj = {'metr' : recall_0}
                
    logging.info('Loading model...')
    ## pkl
    logging.info('  Loading mstd...')
    F = os.path.join(args.model_path, args.mstd_name)
    if not os.path.exists(F):
        msg = 'Model file not available at data-path: {}'
        raise IOError(msg.format(F))        
    with open(F, 'rb') as mstd:
        mean_tr, std_tr = pickle.load(mstd)
    ## h5
    logging.info('  Loading h5...')
    F = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(F):
        msg = 'Model file not available at data-path: {}'
        raise IOError(msg.format(F))    
    model = load_model(F, custom_objects=custom_obj)
    
    # outdir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logging.info('Loading features...')
    x, y, i2n = Utils.load_features_nogt(args.feature_file_table,
                                         force_overwrite=args.force_overwrite,
                                         pickle_only=args.pickle_only,
                                         n_procs=args.n_procs)
    
    logging.info('Loaded {} contigs'.format(len(set(i2n.values()))))    
    n2i = Utils.reverse_dict(i2n)
    x = [xi for xmeta in x for xi in xmeta]
    y = np.concatenate(y)
    
    logging.info('Running model generator...')
    dataGen = Models.Generator(x, y, batch_size=64, shuffle=False, 
                               norm_raw=0, mean_tr=mean_tr, std_tr=std_tr)
    
    logging.info('Computing predictions...')
    scores = Utils.compute_predictions(n2i, dataGen, model,
                                       args.save_path, args.save_name)
    

if __name__ == '__main__':
    pass
