from flask import Blueprint, request, jsonify
from packages.ml_api.api.config import get_logger
from regression_model.predict import make_prediction

from regression_model import __version__ as _version

from packages.ml_api import __version__ as api_version
from packages.ml_api.api.validation import validate_inputs

_logger = get_logger(logger_name=__name__)

import json


prediction_app = Blueprint('prediction_app',__name__)

@prediction_app.route('/health', methods = ['GET'])
def health():
    if request.method=='GET':
        _logger.info("Health status is OK")
        return 'ok'

@prediction_app.route('/version',methods = ['GET'])
def version():
    if request.method == 'GET':
        return jsonify({"model_version":_version,
                        "api_version": api_version})


@prediction_app.route('/v1/predict/regression', methods = ['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()

        input_data, errors = validate_inputs(input_data=json_data)

        input_data = json.dumps(input_data)
        result = make_prediction(input_data=input_data)

        # _logger.info(f'Outputs : {result}')

        prediction = list(result['prediction'])
        version = result.get('version')
        # _logger.info(prediction)
        return jsonify({'prediction':prediction,
                        "'version":version,
                        'errors':errors})
        # return jsonify({'prediction':prediction})