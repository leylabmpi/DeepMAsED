# import
## batteries
import os
import sys
import pytest
import logging
## 3rd party
import numpy as np
## package
from DeepMAsED.Commands import Evaluate as Evaluate_CMD

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Evaluate_CMD.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_evaluate(tmpdir, caplog):
    caplog.set_level(logging.INFO)   
    save_path = tmpdir.mkdir('save_dir')
    args = ['--save-path', str(save_path),
            os.path.join(data_dir, 'n10_r2/feature_files.tsv')]
    args = Evaluate_CMD.parse_args(args)    
    Evaluate_CMD.main(args)

def test_evaluate_not_syn(tmpdir, caplog):
    caplog.set_level(logging.INFO)   
    save_path = tmpdir.mkdir('save_dir')
    args = ['--save-path', str(save_path),
            '--is-synthetic', '0',            
            os.path.join(data_dir, 'n10_r2/feature_files.tsv')]
    args = Evaluate_CMD.parse_args(args)    
    Evaluate_CMD.main(args)

