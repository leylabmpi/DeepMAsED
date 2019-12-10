#!/usr/bin/env python
# import
## batteries
import csv
import os
import _pickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkm_path', default='data', type=str, 
                    help='Where to find checkM csv.')

def main(args):
    if 'Pasolli' in args.checkm_path:
        idx_complete = 7
        idx_contam = 8
        delim = '\t'
        
    else: #if 'Almeida' in args.checkm_path:
        idx_complete = 1
        idx_contam = 2
        delim = '\t'

    
    checkm = {}
    
    with open(args.checkm_path, 'r') as f:
        csv_it = csv.reader(f, delimiter=delim)
        next(csv_it)
    
        for line in csv_it:
            name = line[0].replace('.', '_')
            checkm[name] = {'comp' : float(line[idx_complete].replace(',', '.')), 
                            'cont' : float(line[idx_contam].replace(',', '.'))}
    F = os.path.join('/'.join(args.checkm_path.split('/')[:-1]), 'checkm_parsed.pkl')
    with open(F, 'wb') as f:
        pickle.dump(checkm, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
        

