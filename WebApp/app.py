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
    try:
        r = requests.get('http://127.0.0.1:5001/getinfo').json()
        temp = r['message']
        print(temp)
        return render_template('index.html',temp=temp)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed')
        return "new"

@app.route('/getUserInfo', methods=['GET', 'POST'])
def getUserInfo():
    try:
        url = "http://localhost:5001/getUser"
        data = {'sender': request.form['name']}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(data), headers=headers).json()
        temp = r['sender']
        return render_template('index.html', temp=temp)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed')
        return "new"


if __name__ == '__main__':
    app.run(debug=True)


