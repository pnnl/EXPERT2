# import uqColors
from . import uqColors
from dash import html
import base64
import os

noResultsComp = html.H4("Select to view results.", className='no-output-div')
textVal = [f"Rank {i+1}" for i in range(10)]+['> Rank 10']
textColor = 'black'
colorsInfo = html.Div([
                        html.Div([html.Span([html.P(' '*10, style = {'white-space': 'pre-wrap','background-color':color}),
                                   html.P(text, style={'color': textColor, 'margin-left':'0.5rem'})], style={'display':'flex'})
                                   for text, color in zip(textVal[:4],uqColors.text_colors_rgba[:4] )],
                                   ),
                        html.Div([html.Span([html.P(' '*10, style = {'white-space': 'pre-wrap','background-color':color}),
                                   html.P(text, style={'color': textColor, 'margin-left':'0.5rem'})], style={'display':'flex'})
                                   for text, color in zip(textVal[4:8],uqColors.text_colors_rgba[4:8] )],
                                   ),
                        html.Div([html.Span([html.P(' '*10, style = {'white-space': 'pre-wrap','background-color':color}),
                                   html.P(text, style={'color': textColor, 'margin-left':'0.5rem'})], style={'display':'flex'})
                                   for text, color in zip(textVal[8:],uqColors.text_colors_rgba[8:] )],
                                   )
                                   

 ], style={'display':'flex', 'justifyContent':'space-between','margin-top':'10px'})


def generateBase64Icon(filepath):
    icon= base64.b64encode(open(filepath, 'rb').read()).decode()
    base64Icon = 'data:image/svg+xml;base64,{}'.format(icon)
    return base64Icon

# lowBar = generateBase64Icon('assets/low_prob_bar.svg')
lowBar = generateBase64Icon(os.path.join(os.path.dirname(__file__), 'assets/low_prob_bar.svg'))
mediumBar = generateBase64Icon(os.path.join(os.path.dirname(__file__), 'assets/medium_prob_bar.svg'))
highBar = generateBase64Icon(os.path.join(os.path.dirname(__file__), 'assets/high_prob_bar.svg'))
# lowBar = 'assets/low_prob_bar.png'

