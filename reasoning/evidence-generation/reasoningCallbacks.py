from ast import Raise
from click import style
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash import clientside_callback, ctx
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import dash_core_components as dcc
from dash import html
import pandas as pd
import dash
import json
from dash.exceptions import PreventUpdate
try: from . import Components
except: import Components






def getReasoningCallbacks(app, config, query_type, language_data,source_doc, graph_data = None, graph_text_link = None):

    @app.callback(
    Output("rsn_text_query_header", "children"),
    Output("rsn_text_query_body", "children"),
    Output("source_doc_content", "children"),
    Input("rsn_primary_dd_button", "n_clicks"),
    Input("rsn_reset", "n_clicks"),
    Input('rsn_see_all', 'n_clicks'),
    Input('rsn_sort_dd', 'value'),
    State("rsn_primary_dropdown", "value"),
    prevent_initial_call=True
    )

    def update_options(primary_btn_click, reset_btn_clk, see_all_clk,rsn_sort_val, primary_dd_value): 
        query_icon = 'assets/search-text.svg' if query_type=="text" else 'assets/send-to-graph.svg'

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        see_all = False
        if trigger_id == "rsn_reset":
            return [],'',''
        elif trigger_id == "rsn_see_all" or see_all_clk:
            see_all = True
        try:
            graph_query_ids = list(graph_text_link.query("text_query_id == @primary_dd_value")['graph_query_ids'])[0]
            graph_query_ids = [int(val) for val in graph_query_ids]
            graph_query_df = graph_data.query('graph_query_id == @graph_query_ids')

            # output_graph_dd_options = Components.CompDataLoader.load_dropdown_options(graph_query_df['query'].to_list(),    
            #                                                                         graph_query_df.graph_query_id.to_list())
        except: output_graph_dd_options = []

        text_df = language_data.query('text_query_id==@primary_dd_value')
        # Get the query header 
        text_query_header = [html.P(str(text_df.prompt.to_list()[0]), style={'margin-bottom':'0px'}),
                            html.Img(src=query_icon, style ={'height':'1.5rem', 'width':'1.5rem'})]

        ## get the output query text values
        # print(text_df.answers.to_list()[0])
        output_text_query_ans = Components.CompDataLoader.get_query_outputs(text_df.answers.to_list()[0])
        # docIds = source_doc.document_id.to_list()
        
        
        if query_type == 'text':
            segment_id = text_df['source_documents'].apply(lambda x: [val['segment_id'] for val in x]).to_list()[0]
            filtered_df = source_doc.query('segment_id==@segment_id and full_text != "Not available."')
            source_doc_output = Components.CompDataLoader.get_source_doc_outputs(filtered_df, see_all, rsn_sort_val)
        else:
            source_doc_output = []
        return  text_query_header, output_text_query_ans, source_doc_output
    
   
   
   
    @app.callback(
        Output("rsn_primary_dropdown", "value"),
        Input("rsn_reset", "n_clicks"),
        prevent_initial_call=True
        )

    def update_primary_dd_value(reset_btn_clk):
            return '' 


    @app.callback(
        Output("rsn_source_doc_modal", "is_open"),
        Output("rsn_sd_modal_body", "children"),
        Output("rsn_sd_modal_title", "children"),
        Input({'type': 'expand_button', 'index': ALL}, 'n_clicks'),
        Input('rsn_close_sd_modal', 'n_clicks'),
        prevent_initial_call=True
        )

    def open_modal(segment_id_clk, modal_close):
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "rsn_close_sd_modal" or len(set(segment_id_clk))==1: return False, '', ''
        segment_id = json.loads(trigger_id)
        segment_id = segment_id['index']
        modal_title, modal_body = Components.CompDataLoader.get_modal_output(source_doc, segment_id)
        return True, modal_body,modal_title


    # @app.callback(
    #     Output("rsn_graph_query_header", "children"),
    #     Output("rsn_graph_query_body", "children"),
    #     Input("rsn_secondary_dd_button", "n_clicks"),
    #     Input("rsn_reset", "n_clicks"),
    #     State("rsn_secondary_dropdown", "value"),
    #     prevent_initial_call=True
    #     )

    # def generate_graph_query(btn_click, rest_clk, secondary_dd_value):
    #     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #     if trigger_id == "rsn_reset":
    #         return '',''

    #     graph_df = graph_data.query('graph_query_id == @secondary_dd_value')

    #     # Get the query header 
    #     graph_query_header = html.P(str(graph_df['query'].to_list()[0]), style={'margin-bottom':'0px'})

    #     # get the output query text values
    #     output_graph_query_ans = Components.CompDataLoader.get_query_outputs(graph_df.answers.to_list()[0])
        
    #     return graph_query_header, output_graph_query_ans

    



    