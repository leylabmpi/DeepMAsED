# import
## Batteries
import os
import _pickle as pickle
## 3rd party
import numpy as np
import keras
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import IPython
## Application
from DeepMAsED import Models
from DeepMAsED import Utils


class Config(object):
    def __init__(self, args):
        max_len = args.max_len
        filters = args.filters
        n_conv = args.n_conv
        n_fc = args.n_fc
        n_hid = args.n_hid
        n_features = 11
        pool_window = args.pool_window
        dropout = args.dropout
        lr_init = args.lr_init
        mode = args.mode

def main(args):    
    np.random.seed(12)
    
    # Build model
    config = Config(args)

    deepmased = Models.deepmased(config)
    deepmased.print_summary()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_path = args.save_path

    # Load and process data
    logging.info('Loading data...')
    x, y = Utils.load_features_tr(args.data_path,
                                  max_len=args.max_len,
                                  standard=args.standard,
                                  mode = config.mode, 
                                  pickle_only=args.pickle_only)

    if args.n_folds == -1:
        # Append elements in x
        x = [item for sl in x for item in sl]
        y = np.concatenate(y)


    if args.n_folds > -1:
        if os.path.exists(os.path.join(save_path, str(args.n_folds - 1) + '_model.h5')):
            exit()

        ap_scores = []
        for val_idx in range(args.n_folds):
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
            logging.info('Training network...')
            if config.mode in ['chimera', 'extensive']:
                w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
                class_weight = {0 : 1 , 1: w_one}
                deepmased.net.fit_generator(generator=dataGen, 
                                            validation_data=dataGen_val,
                                            epochs=args.n_epochs, 
                                            use_multiprocessing=True,
                                            verbose=2,
                                            callbacks=[tb_logs, deepmased.reduce_lr])
            elif config.mode == 'edit':
                st = StandardScaler()
                y_tr = st.fit_transform(y_tr)
                y_te = st.transform(y_te)
                deepmased.net.fit(x_tr, y_tr, validation_data=(x_te, y_te),
                                  epochs=args.n_epochs, 
                                  callbacks=[tb_logs, deepmased.reduce_lr])
            logging.info('Computing AUC scores...')
            scores_val = deepmased.predict_generator(dataGen_val)

            ap_scores.append(average_precision_score(y_val[0 : scores_val.size], scores_val))

            deepmased.save(os.path.join(save_path, str(val_idx) + '_model.h5'))

            with open(os.path.join(save_path, 'scores.pkl'), 'wb') as f:
                pickle.dump(ap_scores, f)

    else:
        dataGen = Models.Generator(x, y, args.max_len, batch_size=64,
                                   norm_raw=bool(args.norm_raw))
        deepmased = Models.deepmased(config)
        tb_logs = keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs_final'), 
                                              histogram_freq=0, 
                                              write_graph=True, write_images=True)
        logging.info('Training network...')
        if config.mode in ['chimera', 'extensive']:
            w_one = int(len(np.where(y == 0)[0])  / len(np.where(y == 1)[0]))
            class_weight = {0 : 1 , 1: w_one}
            deepmased.net.fit_generator(generator=dataGen, 
                                        epochs=args.n_epochs, 
                                        use_multiprocessing=True,
                                        verbose=2,
                                        callbacks=[tb_logs, deepmased.reduce_lr])

        logging.info('Saving trained model...')
        outfile = os.path.join(save_path, 'final_model.h5')
        deepmased.save(outfile)
        logging.info('  File written: {}'.format(outfile))

        outfile = os.path.join(save_path, 'mean_std_final_model.pkl')
        with open(outfile, 'wb') as f:
            pickle.dump([dataGen.mean, dataGen.std], f)
        logging.info('  File written: {}'.format(outfile))
            

if __name__ == '__main__':
    pass
        
