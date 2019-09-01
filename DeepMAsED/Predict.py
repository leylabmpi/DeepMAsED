# import
## batteries
import os
## 3rd party
import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import IPython
import _pickle as pickle
## application
from DeepMAsED import Models
from DeepMAsED import Utils


def main(args):
    np.random.seed(12)
    
    if args.cpu_only:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    # Load and process data
    # Provide objective to load
    recall_0 = Utils.class_recall(0)
    recall_1 = Utils.class_recall(1)
    custom_obj = {'metr' : recall_0}
        
    auc = []

    print("Loading model...")
    ## pkl
    F = os.path.join(args.data_path,  'mean_std_final_model.pkl')
    if not os.path.exists(F):
        msg = 'Model file not available at --data_path: {}'
        raise IOError(msg.format(F))        
    with open(F, 'rb') as mstd:
        mean_tr, std_tr = pickle.load(mstd)
    ## h5
    F = os.path.join(args.data_path, 'deepmased.h5')
    if not os.path.exists(F):
        msg = 'Model file not available at --data_path: {}'
        raise IOError(msg.format(F))    
    model = load_model(F, custom_objects=custom_obj)
    
    # outdirs
    #model_path = args.save_path
    outdir = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = os.path.join(args.save_path, 'predictions',
                        os.path.split(args.data_path)[1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading features...")
    args.data_path = os.path.join(args.data_path, 'data')
    x, y, i2n = Utils.load_features_nogt(args.data_path)
    
    print("Loaded {} contigs...".format(len(set(i2n.values()))))    
    n2i = Utils.reverse_dict(i2n)
    x = [xi for xmeta in x for xi in xmeta]
    y = np.concatenate(y)
    
    dataGen = Models.Generator(x, y, batch_size=64,  shuffle=False, 
                               norm_raw=0, mean_tr=mean_tr, std_tr=std_tr)
    
    print("Computing predictions...")
    scores = Utils.compute_predictions(n2i, dataGen, model, outdir)
    

if __name__ == '__main__':
    pass
