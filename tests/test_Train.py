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
