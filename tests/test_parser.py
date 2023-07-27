import pytest
import os
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from parameters_parser import load_args
def test_load_args():
    with pytest.raises(ValueError):
        load_args(None)
    with pytest.raises(ValueError):
        load_args('not_a_file')
    with pytest.raises(ValueError):
        load_args('tests/test_parser.py')
        
    assert load_args('tests/test_config.yml') == {'a': 1, 'b': 2, 'test': 'test'}

