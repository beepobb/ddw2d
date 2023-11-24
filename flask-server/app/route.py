from flask import render_template, url_for
from app import webApp  

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
    title = "Tool"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("tool.html", title=title, url=url)

@webApp.route('/math-analysis')
def math_analysis():
    title = "Math Analysis"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("math-analysis.html", title=title, url=url)

@webApp.route('/history')
def history():
    title = "History"
    url = url_for('static', filename="images/old_landding_bg.png")
    return render_template("history.html", title=title, url=url)