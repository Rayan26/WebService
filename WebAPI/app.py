from flask import Flask, render_template, request
import requests
from flask import jsonify
from flask import json
from flask.json import JSONEncoder

import datacalcul

## Splitting Train/Validation/Test Sets

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

app = Flask(__name__)


@app.route('/action', methods=['GET', 'POST'])
def index():
    global data
    if request.method == 'POST':
        json_request = request.json
        print(json_request)
        if json_request['action'] == 'getUsers':
            if json_request['name'] == 'Alice':
                data = [
                    {
                        "name": "Alice",
                        "lastname": "Lorem",
                        "age": "29"
                    }
                ]
            else:
                if json_request['name'] == 'Bob':
                    data = [{
                        "name": "Bob",
                        "lastname": "Bobino",
                        "age": "21"
                    }]
                else:
                    data = [
                        {
                            "name": "Alice",
                            "lastname": "Lorem",
                            "age": "29"
                        },
                        {
                            "name": "Bob",
                            "lastname": "Bobino",
                            "age": "21"
                        }
                    ]
        if json_request['action'] == 'getWeather':
            url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&APPID=42b9eed7b043ca54cb3231dc33db2c69'
            data = requests.get(url.format(request.form['city_name']))
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/getWeather', methods=['GET', 'POST'])
def getWeather():
    if request.method == 'POST':
        json_request = request.json
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&APPID=42b9eed7b043ca54cb3231dc33db2c69'
        r = requests.get(url.format(json_request[0]['city_name'])).json()
        response = app.response_class(
            response=json.dumps(r),
            status=200,
            mimetype='application/json'
        )
        return response


@app.route('/logistic_regression_imputation', methods=['GET', 'POST'])
def logistic_regression_imputation():
    res = datacalcul.logistic_regression_imputation()
    return jsonify(res)


@app.route('/logistic_regression_dropNA', methods=['GET', 'POST'])
def logistic_regression_dropNA():
    res = datacalcul.logistic_regression_dropNA()
    return jsonify(res)


@app.route('/analyse_linear_LDA', methods=['GET', 'POST'])
def analyse_linear_LDA():
    res = datacalcul.analyse_linear_LDA()
    return jsonify(res)


@app.route('/analyse_linear_QDA', methods=['GET', 'POST'])
def analyse_linear_QDA():
    res = datacalcul.analyse_linear_QDA()
    return jsonify(res)


@app.route('/kernel_ridge', methods=['GET', 'POST'])
def kernel_ridge():
    res = datacalcul.kernel_ridge()
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
