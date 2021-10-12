from flask import Flask, render_template, request
import requests # pip install requests
from flask import jsonify
from flask import json
from flask.json import JSONEncoder

app = Flask(__name__) #Initialise app

# Config



@app.route('/getinfo', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        data = {'sender': 'Alice', 'receiver': 'Bob', 'message': 'We did it!'}
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        return response



if __name__ == '__main__':
    app.run(debug=True, port=5001)
