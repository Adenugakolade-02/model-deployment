
import math

from packages.regression_model.predict import make_prediction
from packages.regression_model.configuration import config
from packages.regression_model.custom_preprocessing_pipelines import model_management

def test_prediction() -> None:
    test_data = model_management.load_data(file_name=config.TESTING_DATA)

    test_json = test_data[0:1].to_json(orient='records')

    prediction  = make_prediction(input_data=test_json)

    #Test 1
    assert prediction is not None
    #Test 2
    assert isinstance(prediction.get('prediction')[0],float)
    #Test 3
    assert math.ceil(prediction.get('prediction')[0]) == 154155

def test_multiple_prediction()->None:
    test_data = model_management.load_data(file_name=config.TESTING_DATA)
    test_data_len = len(test_data)

    test_json = test_data.to_json(orient='records')

    prediction  = make_prediction(input_data=test_json)

    assert prediction is not None
    assert len(prediction.get('prediction')) == 1451
    assert len(prediction.get('prediction')) != test_data_len