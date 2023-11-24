from flask import render_template, url_for, request, jsonify
from app import webApp
import pandas as pd
import os

# landing page
@webApp.route('/')
@webApp.route('/home')
def home():
    title = 'Welcome'
    url = url_for('static', filename="images/landing_bg.png")
    return render_template("home.html", title=title, url=url)

@webApp.route('/about')
def about():
    title = "About"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("about.html", title=title, url=url)

@webApp.route('/ml-model')
def ml_model():
    title = "Machine Learning Model"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("model.html", title=title, url=url)

@webApp.route('/tool')
def tool():
    title = "Tool Prediction Model"
    url = url_for('static', filename="images/BlankBlue.png")
    available_countries = jsonify({'countries': get_available_countries()})
    # print(type(available_countries))
    return render_template("tool.html", title=title, url=url, available_countries=available_countries)

@webApp.route('/math-analysis')
def math_analysis():
    title = "Math Analysis"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("math-analysis.html", title=title, url=url)

@webApp.route('/tool-linear-regression', methods=['POST'])
def run_linear_regression():
    from app.model.tools import Tool
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


@webApp.route('/get_available_countries', methods=['GET'])
def get_available_countries():
    try:
        # Implement logic to get the list of available countries from your DataFrame
        # For example, assuming your DataFrame has a 'Country' column
        csv_filepath = os.path.join(webApp.static_folder, 'csv', 'merged_dataset.csv')
        df = pd.read_csv(csv_filepath)
        countries = df['Country'].unique().tolist()
        print(countries)
        # Return the list of available countries as JSON
        return countries

    except Exception as e:
        # If an error occurs, return an error response as JSON
        return jsonify({'error': str(e)})
    