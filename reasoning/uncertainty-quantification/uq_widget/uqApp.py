from  uqLayout import Layout
import uqCallbacks
# from  .uqLayout import Layout
# from . import uqCallbacks
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px



# Build App
app = Dash(external_stylesheets=[dbc.themes.CERULEAN, '/assets/fontawesome.css', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css'])
# Load layout
layout = Layout()
layout.generateLayout(app)
uqCallbacks.get_callbacks(app)


app.run_server(debug = True, port =8051)