from flask import render_template
from app import webApp  
# landing page
@webApp.route('/')
@webApp.route('/welcome')
def welcome():
    title = "Prediction Model"
    return render_template("welcome.html", title=title)

@webApp.route('/history')
def history():
    title = "History"
    return render_template("history.html", title=title)