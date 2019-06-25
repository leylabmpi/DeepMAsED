import numpy as np
import keras
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
parser.add_argument('--filters', default=8, type=int, 
                    help='N of filters for first conv layer. Then x2.')
parser.add_argument('--n_hid', default=20, type=int, 
                    help='N of units in fully connected layers.')
parser.add_argument('--n_conv', default=2, type=int, 
                    help='N of conv layers.')
parser.add_argument('--n_fc', default=1, type=int, 
                    help='N of fully connected layers.')
parser.add_argument('--n_epochs', default=50, type=int, 
                    help='N of training epochs.')
parser.add_argument('--standard', default=1, type=int, 
                    help='Binary, whether or not to standardize the features.')
parser.add_argument('--max_len', default=10000, type=int, 
                    help='Max contig len, fixed input for CNN.')
parser.add_argument('--dropout', default=0.1, type=float, 
                    help='Rate of dropout.')
parser.add_argument('--pool_window', default=40, type=int, 
                    help='Window size for average pooling.')
parser.add_argument('--test_size', default=0.2, type=float, 
                    help='Size of test set.')
parser.add_argument('--lr_init', default=0.001, type=float, 
                    help='Size of test set.')
parser.add_argument('--mode', default='chimera', type=str, 
                    help='Chimera or edit distance.')
parser.add_argument('--technology', default='megahit', type=str, 
                    help='Megahit or metaspades.')
args = parser.parse_args()

class Config(object):
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

# Build model
config = Config()

chi_net = models.chimera_net(config)
chi_net.print_summary()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

save_path = args.save_path

# Load and process data
print("Loading data...")
x, y = utils.load_features(args.data_path,
                           test_size=args.test_size, 
                           max_len=args.max_len,
                           standard=args.standard,
                            mode = config.mode, 
                            technology=args.technology)

auc_scores = []
for val_idx in range(len(x)):
    x_tr, x_val, y_tr, y_val = utils.kfold(x, y, val_idx)

    chi_net = models.chimera_net(config)

    #Construct generator
    dataGen = models.Generator(x_tr, y_tr, args.max_len, batch_size=64)
    dataGen_val = models.Generator(x_val, y_val, args.max_len, batch_size=64)
    #x_tr, x_val, y_tr, y_val = utils.leave_one_out(x, y, 0, max_len=args.max_len)

    #Train model
    tb_logs = keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'), 
                                         histogram_freq=0, 
                                         write_graph=True, write_images=True)

    print("Training network...")
    if config.mode in ['chimera', 'extensive']:
        w_one = int(len(np.where(y_tr == 0)[0])  / len(np.where(y_tr == 1)[0]))
        class_weight = {0 : 1 , 1: w_one}
        chi_net.net.fit_generator(generator=dataGen, 
                                  validation_data=dataGen_val,
                                  epochs=args.n_epochs, 
                                  use_multiprocessing=True,
                                  verbose=2,
                                  callbacks=[chi_net.reduce_lr])
        #exit()
        #chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=args.n_epochs, 
        #               class_weight=class_weight, 
        #               callbacks=[tb_logs, chi_net.reduce_lr])

    elif config.mode == 'edit':
        st = StandardScaler()
        y_tr = st.fit_transform(y_tr)
        y_te = st.transform(y_te)
        chi_net.net.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=args.n_epochs, 
                       callbacks=[tb_logs, chi_net.reduce_lr])

    print("Computing AUC scores...")
    scores_val = chi_net.predict_generator(dataGen_val)
    auc_scores.append(roc_auc_score(y_val[0 : scores_val.size], scores_val))

    #print("Saving trained model...")
    chi_net.save(os.path.join(save_path, str(val_idx) + '_model.h5'))

with open(os.path.join(save_path, 'scores.pkl'), 'wb') as f:
    pickle.dump(auc_scores, f)



