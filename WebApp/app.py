from flask import Flask, render_template, request
import requests  # pip install requests
from flask.json import JSONEncoder
from flask import jsonify
from flask import json
from pprint import pprint
from json.decoder import JSONDecodeError

app = Flask(__name__)  # Initialise app


# Config


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Appel à un endpoint de notre API qui renvoi des données en dur
@app.route('/getUserInfo', methods=['GET', 'POST'])
def getUserInfo():
    try:
        url = "http://localhost:5001/action"
        data = {'action': 'getUsers', 'name': request.form['name']}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(data), headers=headers).json()
        print(r)
        for i in r:
            name = i["name"]
            lastname = i["lastname"]
            age = i["age"]
        return render_template('index.html', name=name, lastname=lastname, age=age)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed')
        return "new"

# Appel à un endpoint de notre API qui appel l'API d'un service Méteo pour obtenir les informations d'une ville
@app.route('/getWeather', methods=['GET', 'POST'])
def getWeather():
    try:
        url = "http://localhost:5001/getWeather"
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
        return render_template('index.html', temp=temp, weather=weather, min_temp=min_temp, max_temp=max_temp,
                               icon=icon, city_name=request.form['city_name'])

    except ValueError:
        print('Decoding JSON has failed')
        return "new"


if __name__ == '__main__':
    app.run(debug=True)
