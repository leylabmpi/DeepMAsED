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

#def test_predict(tmpdir):
#    out_path = tmpdir.mkdir('save_dir')
#    model_path = os.path.join(data_dir, 'n1000_r3/', 'model')
#    args = [os.path.join(data_dir, 'deepmased_trained'),
#            model_path, '--cpu-only']
#    args = Predict_CMD.parse_args(args)
#    Predict_CMD.main(args)    
    
def test_predict_r3(tmpdir):
    out_path = tmpdir.mkdir('save_dir')
    model_path = os.path.join(data_dir, 'n1000_r3/', 'model')
    args = [os.path.join(data_dir, 'n1000_r3/'),
            '--cpu-only']
    args = Predict_CMD.parse_args(args)
    Predict_CMD.main(args)    
    
