from flask import Flask, render_template, request
import requests  # pip install requests
from flask.json import JSONEncoder
from flask import jsonify
from flask import json
from pprint import pprint
from json.decoder import JSONDecodeError

app = Flask(__name__)  # Initialise app
url_root_api = "http://localhost:5001"

# Config


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/user', methods=['GET', 'POST'])
def user():
    return render_template('user.html')

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    return render_template('weather.html')

@app.route('/get_user', methods=['GET', 'POST'])
def getUserInfo():
    try:
        url = url_root_api + "/action"
        data = {'action': 'getUsers', 'name': request.form['name']}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(data), headers=headers).json()
        print(r)
        for i in r:
            name = i["name"]
            lastname = i["lastname"]
            age = i["age"]
        return render_template('user.html', name=name, lastname=lastname, age=age)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed')
        return "new"

# Appel à un endpoint de notre API qui appel l'API d'un service Méteo pour obtenir les informations d'une ville
@app.route('/get_weather', methods=['GET', 'POST'])
def getWeather():
    try:
        url = url_root_api + "/getWeather"
        data = [{'city_name': request.form['city_name']}]
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        response = requests.post(url, data=json.dumps(data), headers=headers).json()
        print(response)
        temp = response['main']['temp']
        weather = response['weather'][0]['description']
        min_temp = response['main']['temp_min']
        max_temp = response['main']['temp_max']
        icon = response['weather'][0]['icon']
        print(temp, weather, min_temp, max_temp, icon)
        return render_template('weather.html', temp=temp, weather=weather, min_temp=min_temp, max_temp=max_temp,
                               icon=icon, city_name=request.form['city_name'])

    except ValueError:
        print('Decoding JSON has failed')
        return "new"

def logistic_regression_dropNA():
    url = url_root_api + "/logistic_regression_dropNA"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url,  headers=headers).json()
    return response

def analyse_linear_LDA():
    url = url_root_api + "/analyse_linear_LDA"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url,  headers=headers).json()
    return response

def analyse_linear_QDA():
    url = url_root_api + "/analyse_linear_QDA"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url,  headers=headers).json()
    return response

def kernel_ridge():
    url = url_root_api + "/kernel_ridge"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url,  headers=headers).json()
    return response




@app.route('/getRoute', methods=['GET', 'POST'])
def getRoute():
    LR = request.args.get("LR", default=None, type=str)
    LDA = request.args.get("LDA", default=None, type=str)
    QDA = request.args.get("QDA", default=None, type=str)
    KR = request.args.get("KR", default=None, type=str)

    response = None
    if (LR) :
        response = logistic_regression_dropNA()
    elif (LDA):
        response = analyse_linear_LDA()
    elif (QDA):
        response = analyse_linear_QDA()
    elif (KR):
        response = kernel_ridge()


    return render_template('index.html',  response=response)



if __name__ == '__main__':
    app.run(debug=True)
