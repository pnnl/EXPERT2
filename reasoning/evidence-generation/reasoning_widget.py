import reasoningLayout
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import plotly.express as px
import socket
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import reasoningLayout, reasoningCallbacks


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_new_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]

class LoadWidget:
    def __init__(self, query_type, language_data_filepath, source_doc_filepath = None, graph_data_filepath = None, graph_text_link_filepath = None):
        try:            
            # Build Widget
            app = JupyterDash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css'])
            
           # Load Layout and Data
            config = None

            # graph_data = pd.read_json(graph_data_filepath, lines = True)
            language_data = pd.read_json(language_data_filepath, lines = True)
            # graph_text_link = pd.read_json(graph_text_link_filepath, lines = True)
            source_doc = pd.read_json(source_doc_filepath, lines = True) if query_type == 'text' else None
            layout = reasoningLayout.get_layout(config, query_type, language_data,source_doc, graph_data = None, graph_text_link = None )
            app.layout = layout
            reasoningCallbacks.getReasoningCallbacks(app, config, query_type, language_data,source_doc, graph_data = None, graph_text_link = None)
            
            # Run Widget
            port = 8050

            if is_port_in_use(port): port = get_new_port()
            print(f"(User Message: If running this widget on a virtual machine, port forward {port} to your local machine and then run the widget)")
            del app.config._read_only["requests_pathname_prefix"]
            app.run_server(port = port, mode ='inline', debug = True)
        except Exception as e: print(f"Error Loading Data...\n{str(e)}")