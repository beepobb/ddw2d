from flask import Flask
from flask_bootstrap import Bootstrap5
import os


webApp = Flask(__name__, static_url_path='/flask-server/app/static')
bootstrap = Bootstrap5(webApp)

# error handling
if os.environ.get('FLASK_ENV') == 'production':
    print("Running in production environment.")
else:
    print(f"Running in {os.environ.get('FLASK_ENV', 'development')} environment.")

from app import route

if __name__ == '__main__':
    webApp.run(debug=True)