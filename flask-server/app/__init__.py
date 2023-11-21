from flask import Flask
from flask_bootstrap import Bootstrap5

webApp = Flask(__name__, static_folder='static')
bootstrap = Bootstrap5(webApp)

from app import route

if __name__ == '__main__':
    webApp.run(debug=True)