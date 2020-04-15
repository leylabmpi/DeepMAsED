# import
## Batteries
import os
import sys
import logging
import _pickle as pickle
## 3rd party
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import IPython
## Application
from DeepMAsED import Models
from DeepMAsED import Utils


class Config(object):
    def __init__(self, args):
        self.max_len = args.max_len
        self.filters = args.filters
        self.n_conv = args.n_conv
        self.n_fc = args.n_fc
        self.n_hid = args.n_hid
        self.n_features = 11
        self.pool_window = args.pool_window
        self.dropout = args.dropout
        self.lr_init = args.lr_init

def main(args):
    # init
    np.random.seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_path = args.save_path
    
    # Build model
    config = Config(args)
    if not args.pickle_only:
        logging.info('Building model')
        deepmased = Models.deepmased(config)
        deepmased.print_summary()

    # Load and process data
    x, y = Utils.load_features_tr(args.feature_file_table,
                                  max_len=args.max_len,
                                  technology = args.technology,
                                  pickle_only = args.pickle_only,
                                  force_overwrite = args.force_overwrite,
                                  n_procs = args.n_procs)

    # kfold cross validation
    if args.n_folds >= 0:
        logging.info('Running kfold cross validation. n-folds: {}'.format(args.n_folds))
        outfile_h5 = os.path.join(save_path, str(args.n_folds - 1) + '_model.h5')
        if os.path.exists(outfile_h5) and args.force_overwrite is False:
            msg = 'Output already exists ({}). Use --force-overwrite to overwrite the file'
            raise IOError(msg.format(outfile_h5))

        # iter over folds
        ap_scores = []
        for val_idx in range(args.n_folds):
            logging.info('Fold {}: Constructing model...'.format(val_idx))        
            x_tr, x_val, y_tr, y_val = Utils.kfold(x, y, val_idx, k=args.n_folds)
            deepmased = Models.deepmased(config)

            #Construct generator
            dataGen = Models.Generator(x_tr, y_tr, args.max_len,
                                       batch_size=64, norm_raw=bool(args.norm_raw))

            # Init validation generator and 
            dataGen_val = Models.Generator(x_val, y_val, args.max_len, batch_size=64, 
                                           shuffle=False, norm_raw=bool(args.norm_raw), 
                                           mean_tr=dataGen.mean, std_tr=dataGen.std)

            #Train model
            tb_logs = keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'), 
                                                  histogram_freq=0, 
                                                  write_graph=True, write_images=True)
            logging.info('Fold {}: Training network...'.format(val_idx))
            ## binary classification (extensive misassembly)
            try:
                w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
            except ZeroDivisionError:
                logging.warning('  No misassemblies present!')
                w_one = 0
            class_weight = {0 : 1 , 1: w_one}
            deepmased.net.fit_generator(generator=dataGen, 
                                        validation_data=dataGen_val,
                                        epochs=args.n_epochs, 
                                        use_multiprocessing=args.n_procs > 1,
                                        workers=args.n_procs,
                                        verbose=2,
                                        callbacks=[tb_logs, deepmased.reduce_lr])
            # AUC scores
            logging.info('Fold {}: Computing AUC scores...'.format(val_idx))
            scores_val = deepmased.predict_generator(dataGen_val)
            ap_scores.append(average_precision_score(y_val[0 : scores_val.size], scores_val))

            # Saving data
            outfile_h5_fold = os.path.join(save_path, str(val_idx) + '_model.h5')
            deepmased.save(outfile_h5_fold)
            logging.info('Fold {}: File written: {}'.format(val_idx, outfile_h5_fold))
            outfile_pkl_fold = os.path.join(save_path, 'scores.pkl')
            with open(outfile_pkl_fold, 'wb') as f:
                pickle.dump(ap_scores, f)
            logging.info('Fold {}: File written: {}'.format(val_idx, outfile_pkl_fold))

    else:
        # Skip kfold and simply pool all the data for training
        ## all elements in x and y are combined
        logging.info('NOTE: Training on all pooled data!')
        x = [item for sl in x for item in sl]
        y = np.concatenate(y)
        
#         #downsample to half
#         import random
#         dwnsample = np.array(random.sample(range(len(y)), int(len(y)/2)))
#         x = np.array(x)[dwnsample]
#         y = np.array(y)[dwnsample]

        logging.info('Constructing model...')
        dataGen = Models.Generator(x, y, args.max_len, batch_size=64,
                                   norm_raw=bool(args.norm_raw))
        deepmased = Models.deepmased(config)
        tb_logs = keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs_final'), 
                                              histogram_freq=0, 
                                              write_graph=True, write_images=True)
        
        logging.info('Training network...')
        deepmased.net.fit_generator(generator=dataGen,
                                    epochs=args.n_epochs, 
                                    use_multiprocessing=args.n_procs > 1,
                                    workers=args.n_procs,
                                    verbose=2,
                                    callbacks=[tb_logs, deepmased.reduce_lr])
            
        logging.info('Saving trained model...')
        x = [args.save_name, args.technology, 'model.h5']
        outfile = os.path.join(save_path, '_'.join(x))
        deepmased.save(outfile)
        logging.info('  File written: {}'.format(outfile))        
        x = [args.save_name, args.technology, 'mean_std.pkl']
        outfile = os.path.join(save_path, '_'.join(x))
        with open(outfile, 'wb') as f:
            pickle.dump([dataGen.mean, dataGen.std], f)
        logging.info('  File written: {}'.format(outfile))
            

if __name__ == '__main__':
    pass
        
