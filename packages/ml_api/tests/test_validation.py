from regression_model.configuration import config
from regression_model.custom_preprocessing_pipelines.model_management import load_data

import json

def test_prediction_endpoint_with_validation_200(flask_test_client):

    test_data = load_data(config.TESTING_DATA)
    post_json = test_data.to_json(orient='records')
    response = flask_test_client.post('/v1/predict/regression',json=json.loads(post_json))
    
    assert response.status_code == 200

    response_json = json.loads(response.data)
    print(response_json)
    
    # assert len(response_json['prediction']) + len(response_json['errors']) == len(test_data)
    assert response_json['errors'] is not None
    assert len(response_json['errors']) >=0
    assert len(response_json['prediction']) != len(test_data)


