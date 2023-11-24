from flask import render_template, url_for
from app import webApp  

# landing page
@webApp.route('/')
@webApp.route('/landing')
def landing():
    title = 'Welcome'
    url = url_for('static', filename="images/landing_bg.png")
    return render_template("landing.html", title=title, url=url)

@webApp.route('/welcome')
def welcome():
    title = "Prediction Model"
    url = url_for('static', filename="images/old_landing_bg.png")
    return render_template("welcome.html", title=title, url=url)

@webApp.route('/history')
def history():
    title = "History"
    url = url_for('static', filename="images/BlankBlue.png")
    return render_template("history.html", title=title, url=url)