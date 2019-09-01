# import
## batteries
import os
import sys
import pytest
## 3rd party
import numpy as np
## package
from DeepMAsED.Commands import Predict as Predict_CMD

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')


def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Predict_CMD.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_predict(tmp_path):
    indir = os.path.join(data_dir, 'deepmased_trained')
    outdir = os.path.join(tmp_path, 'predict_cpu')
    args = ['--cpu_only', '--data_path', indir, '--save_path', outdir]
    args = Predict_CMD.parse_args(args)
    Predict_CMD.main(args)
    F = os.path.join(outdir,'predictions', 'deepmased_trained', 'predictions.csv')
    assert os.path.isfile(F)
    
