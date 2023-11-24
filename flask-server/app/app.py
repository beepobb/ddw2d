from flask import Flask, render_template, request, jsonify
from tools import Tool  # Import your Tool class

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_linear_regression', methods=['POST'])
def run_linear_regression():
    try:
        data = request.json  # Get user input from the request
        country = data['country']
        money = data['money']

        # Create an instance of the Tool class
        tool_instance = Tool()

        # Set user input in the instance
        tool_instance.country = country
        tool_instance.money = money

        # Run linear regression and get the prediction
        prediction = tool_instance.predict(tool_instance.country, tool_instance.money)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

