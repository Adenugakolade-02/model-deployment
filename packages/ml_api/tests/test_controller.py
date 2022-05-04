from regression_model.configuration import config
from regression_model.custom_preprocessing_pipelines.model_management import load_data

from ml_api.api import __version__ as _api_version

import json
import math



def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200

def test_version_endpoint_return_version(flask_test_client):
    response = flask_test_client.get('/version')
    response_json = json.loads(response.data)

    assert response.status_code == 200
    assert response_json['api_version'] == _api_version  

def test_prediction_endpoints_return_prediction(flask_test_client):
    test_data = load_data(file_name=config.TESTING_DATA)
    test_len = len(test_data)
    post_json = test_data.to_json(orient='records')
    # print(post_json)

    response = flask_test_client.post('/v1/predict/regression',json = post_json)

    ###Test
    assert response.status_code == 200
    
    response_json = json.loads(response.data)
    prediction = response_json['prediction']

    assert prediction is not None
    assert len(prediction) != test_len


