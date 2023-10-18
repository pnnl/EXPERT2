# import uqColors, uqDataProcessor, uqComponents
from . import uqColors, uqDataProcessor, uqComponents
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_daq as daq


class Layout:
    def __init__(self):
        return
    def generateLayout(self, app, data):
        app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([dbc.Col(
                                    html.Div([
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.Div([
                                                    html.H4("Uncertainty Quantification", style={'color':'#343434'}), 
                                                     html.I(id = 'main_info', className="fa fa-info-circle fa-lg", 
                                                            style={'line-height':'30px', 'margin-left':'5px'}), 
                                                    dbc.Tooltip(uqComponents.colorsInfo, 
                                                                target = 'main_info', 
                                                                placement = 'bottom', 
                                                               className ='main-tooltip', 
                                                               style={'background-color':'white', 
                                                               'border':'1px solid black'})                                                
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
                                        dcc.Dropdown(
                                            id = 'model_dropdown',
                                            options = data['modelOptions'],
                                            value="model_0",
                                            placeholder = "Select Model...",
                                            optionHeight=40
                                                   )                                                
                                            ], 
                                    style={'textAlign': 'left'}) 
                                        ], style={'padding-bottom':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=4, style={'display':'none'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        dcc.Dropdown(
                                            id = 'genAlg_dropdown',
                                            options = data['genAlgoOptions'],
                                            placeholder = "Select Generator Algorithm...",
                                            optionHeight=40
                                                   )                                                
                                            ], 
                                    style={'textAlign': 'left'}) 
                                        ], style={'padding-bottom':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=6),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        dcc.Dropdown(
                                            id = 'uncert_est_dropdown',
                                            options = data['uncertainityEstimatorOptions'],
                                            placeholder = "Select Uncertainty Estimator...",
                                            optionHeight=40
                                                   )                                                
                                            ], 
                                    style={'textAlign': 'left'}) 
                                        ], style={'padding-bottom':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=6),                       
                    ], align='top'), 
                    
                     dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        dcc.Dropdown(
                                            id = 'question_dropdown',
                                            options = data['questionOptions'],
                                            placeholder = "Select Question...",
                                            optionHeight=40
                                                   )                                   
                                            ], 
                                    style={'textAlign': 'left'}) 
                                        ]),
                                style={'border':'none'}
                                ),
                        ], width=10),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        dbc.Button("GO", id = 'output_button', color="primary", 
                                        disabled = True,
                                        className='go-button-active',
                                                ),
                                            ], 
                                    style={'textAlign': 'left'}) 
                                        ]),
                                style={'border':'none'}
                                ),
                        ], width=2),
                        
                    ], align='top'), 

                     dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                              html.Div([
                                              html.Div(
                                                        id = 'output_text_comp',
                                                        children = 'Loading ...',
                                                       className='text-output-div',
                                                        ),
                                        html.Button("Show More", id = 'more_text_button', 
                                        className = 'more-text-button'
                                                ),]),                      
                                            ], 
                                    className='text-output-main',
                                   ) 
                                        ], style={'padding-bottom':'0px','padding-top':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=12),
                    ], align='top'), 

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        html.H5('...', style={'color':'black', 'line-height':'80px'})
                                            ], id = 'uncert_estimator_val',
                                    className='estimator-output', style={'color':'black', 'height':'90px'}
                                   ) 
                                        ], style={'padding-top':'10px','padding-bottom':'0px'}),
                                style={'border':'none'}
                                ),
                        ], width=12),
                    ], align='center'), 
                ]), 
            className = 'main-card')
        ], style={'padding':"10px", "padding-bottom":'5px'})

