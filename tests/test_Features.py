# import
## batteries
import os
import sys
import pytest
import logging
## 3rd party
import numpy as np
## package
from DeepMAsED.Commands import Features as Features_CMD

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Features_CMD.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_predict_default_model(tmpdir, caplog):
    caplog.set_level(logging.INFO)   
    outdir = tmpdir.mkdir('outdir')
    args = ['--outdir', str(outdir), '--debug',
            os.path.join(data_dir, 'n4_feat/bam_fasta.tsv')]
    args = Features_CMD.parse_args(args)
    Features_CMD.main(args)  
    
