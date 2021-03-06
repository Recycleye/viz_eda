"""dash_app.py: creating the dash app and the cache."""
import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

external_stylesheets = [
    # Loading screen CSS
    # dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/brPBPO.css', ]
    dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config["suppress_callback_exceptions"] = True

# ==== Set up Cache ==== #
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
