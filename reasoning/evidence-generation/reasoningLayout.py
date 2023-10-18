# import uqColors, uqDataProcessor, uqComponents
# from . import uqColors, uqDataProcessor, uqComponents
from sys import displayhook
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_daq as daq
try: from . import Components
except: import Components


def get_layout(config,query_type, language_data,source_doc, graph_data = None, graph_text_link = None):
    compDefaultSelGen = Components.CompDefaultSelGen(graph_data, language_data, graph_text_link)
    widget_title = "Text Based Reasoning" if query_type=="text" else "Graph Based Reasoning"
    query_container_width = 6 if query_type=="text" else 12
    source_doc_display = '' if query_type=="text" else 'none'
    query_text = "Text" if query_type=="text" else "Graph"

    layout = html.Div([
        dbc.Card(
            dbc.CardBody([
                dbc.Row([dbc.Col(
                                html.Div([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div([
                                                html.H4(widget_title, style={'color':'#343434'}), 
                                                #  dbc.Button(id = 'main_info', className="bi bi-list", 
                                                #         style={'line-height':'30px', 'margin-left':'5px'}),                                                
                                                    ], style={'textAlign': 'left', 'display':'flex'}) 
                                                    ],style = {'padding-top':'0px', 'padding-bottom':'0px'}),
                                                    style={'border':'none'}
                                            ),
                                ])
                )]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                            
                                        ], 
                                style={'textAlign': 'left'}) 
                                    ], style={'padding-bottom':'0px'}),
                            style={'border':'none'}
                            ),
                    ], width=4, style={'display':'none'}),
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([dbc.Row([
                                    dbc.Col(dcc.Dropdown(id = 'rsn_primary_dropdown', 
                                                        className = 'bottom-border-dropdown',
                                                        clearable=True,
                                                        placeholder = "Start typing to ask question...",
                                                        options = compDefaultSelGen.primary_dropdown_options), width = 11, style={'paddig':'0px'}),
                                            dbc.Col(dbc.Button(id = 'rsn_primary_dd_button', className="fa fa-arrow-circle-o-right fa-lg", 
                                                        style={    'background-image': 'none', 'background-color': 'rgba(0,0,0,0)',
                                                                    'border': 'none','color': '#6aa2dd', 'font-size':'2.4rem', 'padding-left':'0px'}), 
                                                                    width=1, style={'paddig':'0px'}),

                                        ])        
                                        ], 
                                style={'textAlign': 'left'}) 
                                    ], style={'padding-bottom':'0px'}),
                            style={'border':'none'}
                            ),
                    ], width=12),                     
                ], align='top', style={'margin-bottom':'40px'}), 
                

                dbc.Row([
                    dbc.Col([
                            dbc.Row([
                                dbc.Col([

                                ], width=12),
                            ], align='top'), 

                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                                dbc.CardBody([
                                    html.Div(f"{query_text} Query Log", style ={'margin-bottom':'10px', 'font-size':'1.1rem'}),
                                    dbc.Button(" Reset",id = "rsn_reset", className = 'fa fa-repeat', style={'background-color':'rgba(0,0,0,0)',
                                        'color':'black', 'border':'none' })
                                        ], style={'display':'flex','justifyContent':'space-between',
                                        'padding-bottom':'5px','padding-top':'0px'}),
                                style={'border':'none'}
                                ),

                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    dbc.Row([dbc.Card([
                                                        dbc.CardHeader(id = 'rsn_text_query_header', 
                                                                        style ={'border':'none', 'height':'50px',
                                                                                'display':'flex', 'align-items':'center','justifyContent':'space-between'}),
                                                        dbc.CardBody( id = 'rsn_text_query_body', 
                                                                        className='text-output-main-top', 
                                                                        style={'margin-bottom':'0px'})
                    
                                    ], style ={"border":'none', 'padding-right':'0px', 'padding-left':'0px', 'width':'100%', 
                                            "margin-right":'0.1rem'}) ], ), # "margin-left":'0.2rem',
                                    # dbc.Row([
                                    #         dbc.Col(dcc.Dropdown(id = 'rsn_secondary_dropdown', 
                                    #                         disabled= True,
                                    #                         className = 'bottom-border-dropdown',
                                    #                         clearable=True,
                                    #                         placeholder = "Compliment your query. Select the drop-down...",
                                    #                         ), width = 11, id= 'col_sec'),
                                    #         dbc.Col(dbc.Button(id = 'rsn_secondary_dd_button', className="fa fa-arrow-circle-o-right fa-lg", 
                                    #                         style={ 'background-image': 'none', 'background-color': 'rgba(0,0,0,0)',
                                    #                                     'border': 'none','color': '#6aa2dd', 'font-size':'2.4rem', 'padding-left':'0px',
                                    #                                     }, disabled = True), 
                                    #                 width=1, style={'padding':'0px'}),

                                    #         ], style={'padding-left':'10px', 'height':'49px', 
                                    #                     'padding-top':'0px','padding-bottom':'0px'})  
                                ], 
                            ) 
                                    ], className = 'text-output-wrap',style={'padding-bottom':'0px','padding-top':'0px',
                                    'min-height': '634px', 'max-height': '634px'}, ),
                            style={'border':'none', 'padding-right':'15px', 'padding-left':'15px'}
                            ),
                    ], width=12),
                ], align='center', id = 'primary_result_query'), 

                # dbc.Row([
                #     dbc.Col([
                #         dbc.Card(
                #             dbc.CardBody([
                #                 html.Div([
                #                     dbc.Row([dbc.Card([
                #                                         dbc.CardHeader(id = 'rsn_graph_query_header', 
                #                                                         style ={'border':'none', 'height':'50px',
                #                                                                 'display':'flex', 'align-items':'center'}),
                #                                         dbc.CardBody( id = 'rsn_graph_query_body', 
                #                                                         className='text-output-main-top', 
                #                                                         style={'margin-bottom':'0px', 'min-height':'250px'})
                    
                #                     ], style ={"border":'none', 'padding-right':'0px', 'padding-left':'0px', 'width':'100%',
                #                                 "margin-right":'0.1rem' # "margin-left":'0.2rem',
                #                     }) ]),
                #                 ], 
                #             ) 
                #                     ], className = 'text-output-wrap',style={'padding-bottom':'0px','padding-top':'0px', 
                #                                                                 }, ),
                #             style={'border':'none', 'padding-right':'15px', 'padding-left':'15px'}
                #             ),
                #     ], width=12),
                # ], align='center',  id = 'secondary_result_query', style={'margin-top':'34px'}), 

                                ], width = query_container_width), 

            dbc.Col([ dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Stack([html.P("Source Documents", style = {'padding':'0px', 'margin':'0px'}, ), 
                                            dbc.Button("See All",id = "rsn_see_all",className ="ms-auto", style={'background-image':'none', 'border':'none', 'color':'#0d6efd', 'background-color':'rgba(0,0,0,0)'}), 
                                            dcc.Dropdown(options = [{'label':'Most Recent', 'value':'most_recent'},
                                                                    {'label':'Oldest', 'value':'oldest'}
                                                                    ],
                                                        value = 'most_recent', 
                                                        clearable = False,
                                                        id = 'rsn_sort_dd', 
                                                        style ={'width':'9vw', 'border-radius':'20px'},
                                                        
                                            )], gap =2, direction ='horizontal', style={'margin-bottom':'5px'}),
                                    html.Div([], id = 'source_doc_content', 
                                    className='text-output-main', style={'min-height': '634px', 'max-height': '634px', 'padding':'10px'}
                                ) 
                                        ], style={'padding-bottom':'0px','padding-top':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=12),
                    ], align='center') ], width = 6, id = 'source_documents',style={'display':source_doc_display}),

                    # MODAL
                            dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle("Modal with scrollable body", id = 'rsn_sd_modal_title')),
                                            dbc.ModalBody("text", id = 'rsn_sd_modal_body', ),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    "Close",
                                                    id="rsn_close_sd_modal",
                                                    className="ms-auto",
                                                    n_clicks=0, style={'background-image':'none','background-color':'#6aa3dd'}
                                                )
                                            ),
                                        ],
                                        id="rsn_source_doc_modal",
                                        scrollable=True,
                                        centered=True,
                                        is_open=False,
                                        style ={'height':'80%', 'margin-bottom':'10%'}
                                        
                                    ),

                ]),

            ]), 
        className = 'main-card')
    ], style={'padding':"10px", "padding-bottom":'5px'})

    return layout

