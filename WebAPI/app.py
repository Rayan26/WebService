from flask import Flask, render_template, request
import requests # pip install requests
from flask.json import JSONEncoder

app = Flask(__name__) #Initialise app

# Config



@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        information = 'une information'


        
        print(temp,weather,min_temp,max_temp,icon)
        return render_template('index.html',temp=temp,weather=weather,min_temp=min_temp,max_temp=max_temp,icon=icon, city_name = city_name)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
