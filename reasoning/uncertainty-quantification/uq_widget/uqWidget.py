# from  uqLayout import Layout
# import uqCallbacks, uqConfig
from  .uqLayout import Layout
from . import uqCallbacks, uqConfig
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import plotly.express as px
import socket
import warnings
warnings.filterwarnings('ignore')

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_new_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]

class LoadWidget:
    def __init__(self, data_path):
        try:
            # Process data 
            data = uqConfig.processData(data_path)
            
            # Build Widget
            app = JupyterDash(__name__,external_stylesheets=[dbc.themes.CERULEAN,'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css'])
            
            # Load Layout
            layout = Layout()
            layout.generateLayout(app, data)
            uqCallbacks.get_callbacks(app, data)
            
            # Run Widget
            port = 8050

            if is_port_in_use(port): port = get_new_port()
            print(f"(User Message: If running this widget on a virtual machine, port forward {port} to your local machine and then run the widget)")
            app.run_server(port = port, mode ='inline',debug = True)
        except Exception as e: print(f"Error Loading Data...\n{str(e)}")
           
