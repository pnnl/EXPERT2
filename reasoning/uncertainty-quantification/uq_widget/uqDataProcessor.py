# import uqComponents
# import uqColors
from . import uqComponents
from . import uqColors
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
import numpy as np


def generateTextOutput(model, genAlgo, question, uncert_estimator, data):

    filterParams =(data['modelDict'][model], data['genAlgoDict'][genAlgo], data['questionDict'][question])

    df = data['df_raw'].get_group(filterParams)
    outputTextListOfList = df.Token_List.to_list()
    outputColorListOfList = df.Color_List.to_list()
    hoverListOfList = df.Hover_List.to_list()

    resultList = []
    for outputTextList, outputColorList, hoverList in zip(outputTextListOfList, outputColorListOfList, hoverListOfList):
        individualResultList = []
        first_word = True
        for text, color, hover in zip(outputTextList, outputColorList, hoverList):
            if text[0] == ' ':
                individualResultList.append( html.P(' ', 
                                        style={'white-space': 'pre-wrap', 'margin-bottom':'0px'})
                                )
                individualResultList.append( html.P(text[1:], 
                                                title = hover,  
                                                style={'background-color':color, 'white-space': 'pre-wrap', 'margin-bottom':'0px'})
                                        )

            else:
                if first_word:  individualResultList.append(html.P(' ', 
                                        style={'white-space': 'pre-wrap', 'margin-bottom':'0px'}))
                individualResultList.append( html.P(text, 
                                        title = hover,  
                                        style={'background-color':color, 'margin-bottom':'0px'})
                            )
            first_word = False
        resultList.append(individualResultList)
        
        
          
    outputTextComponent = [dbc.Row(res, className='row-text-div') for res in resultList]
    uncertVal = df[uncert_estimator].unique()[0] 
    outputUncerEstResult = str(uncertVal)
    outputUncerEstResult = f"Uncertainty Estimation: {outputUncerEstResult[:5]}"

    # Condition to classify low, med, high probabilities
    if uncertVal < 0.5: compBar = uqComponents.lowBar
    elif uncertVal >=0.5 and uncertVal <= 0.75: compBar = uqComponents.mediumBar
    else: compBar = uqComponents.highBar


    uncertComp = html.Div([html.H5(outputUncerEstResult, style={'color':'black'}), 
                          html.Img(src = compBar),     
                          ], style={'margin-top':'10px'})
    return outputTextComponent, uncertComp