# import
## batteries
import os
import sys
import pytest
import logging
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
    
def test_predict_default_model(tmpdir, caplog):
    caplog.set_level(logging.INFO)   
    save_path = tmpdir.mkdir('save_dir')
    args = ['--cpu-only',
            '--save-path', str(save_path),
            os.path.join(data_dir, 'n10_r2/feature_files.tsv')]
    args = Predict_CMD.parse_args(args)
    Predict_CMD.main(args)  
    
