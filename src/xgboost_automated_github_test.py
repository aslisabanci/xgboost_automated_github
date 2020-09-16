from . import xgboost_automated_github

def test_xgboost_automated_github():
    assert xgboost_automated_github.apply("Jane") == "hello Jane"
