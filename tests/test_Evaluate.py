# import
## batteries
import os
import sys
import pytest
## 3rd party
import numpy as np
## package
from DeepMAsED import Evaluate
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

def test_evaluate_r3(tmpdir):
    save_path = tmpdir.mkdir('save_dir')
    args = [os.path.join(data_dir, 'n1000_r3/'),
            os.path.join(data_dir, 'n1000_r3/', 'model')]
    args = Evaluate_CMD.parse_args(args)
    Evaluate.main(args)

def test_evaluate_r3_not_syn(tmpdir):
    save_path = tmpdir.mkdir('save_dir')
    args = [os.path.join(data_dir, 'n1000_r3/'),
            os.path.join(data_dir, 'n1000_r3/', 'model'),
            '--is-synthetic', '0']
    args = Evaluate_CMD.parse_args(args)
    Evaluate.main(args)
