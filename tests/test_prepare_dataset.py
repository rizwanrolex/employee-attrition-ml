import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_dataset import Xorshift32, HEADER

def test_rng_is_repeatable():
    a=Xorshift32(20261111)
    b=Xorshift32(20261111)
    assert [a.random() for _ in range(20)] == [b.random() for _ in range(20)]

def test_columns_present():
    assert len(HEADER)==10 and HEADER[-1]=='Attrition'
