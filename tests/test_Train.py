# import
## batteries
import os
import sys
import pytest
## 3rd party
import numpy as np
## package
from DeepMAsED.Commands import Train as Train_CMD

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Train_CMD.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_train_r3_pkl_only(tmpdir):
    save_path = tmpdir.mkdir('save_dir')
    args = ['--n-folds', '3', '--n-epochs', '2', '--pickle-only',
            '--save-path', str(save_path),
            os.path.join(data_dir, 'n1000_r3/')]
    args = Train_CMD.parse_args(args)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Train_CMD.main(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
    
def test_train_r3(tmpdir):
    save_path = tmpdir.mkdir('save_dir')
    args = ['--n-folds', '3', '--n-epochs', '2',
            '--save-path', str(save_path),
            os.path.join(data_dir, 'n1000_r3/')]
    args = Train_CMD.parse_args(args)
    Train_CMD.main(args)

