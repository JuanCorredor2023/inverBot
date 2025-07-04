from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about_us')
def about_us():
    return render_template('about_us.html')

@main.route('/demo')
def demo():
    return render_template('demo.html')

