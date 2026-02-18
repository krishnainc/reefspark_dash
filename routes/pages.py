"""
Page routes - rendering HTML templates
"""
from flask import Blueprint, render_template

pages = Blueprint('pages', __name__)


@pages.route('/')
def home():
    return render_template('index.html')


@pages.route('/litters')
def litters_page():
    return render_template('litters.html')


@pages.route('/simek')
def simek_page():
    return render_template('simek.html')


@pages.route('/live-monitor')
def live_monitor():
    return render_template('live_monitor.html')


@pages.route('/forecast')
def forecast_page():
    return render_template('forecast.html')
