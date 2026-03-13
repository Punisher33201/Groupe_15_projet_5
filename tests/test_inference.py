import pytest
import pandas as pd
from src.inference import InferenceEngine

def test_inference_engine_init():
    engine = InferenceEngine()
    assert hasattr(engine, 'scaler')
    assert hasattr(engine, 'models')
    assert 'SVC' in engine.models
    assert 'RandomForest' in engine.models

def test_predict():
    engine = InferenceEngine()
    input_dict = {col: 1.0 for col in engine.cols_used}
    for model_name in engine.models.keys():
        proba = engine.predict(input_dict, model_name)
        assert isinstance(proba, float)
        assert proba >= 0 and proba <= 1
    