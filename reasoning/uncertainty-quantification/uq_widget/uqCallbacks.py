# import uqComponents, uqDataProcessor
from . import uqComponents, uqDataProcessor
from dash import dcc, html, Input, Output, State, callback_context



def get_callbacks(app, data):

    @app.callback(
        Output('output_text_comp', 'children'),
        Output('uncert_estimator_val', 'children'),
        Output('more_text_button', 'style'),
        Input('output_button', 'n_clicks'),
        Input('more_text_button', 'n_clicks'),
        State('uncert_est_dropdown', 'value'),    
        State('question_dropdown', 'value'),
        State('genAlg_dropdown', 'value'),
        State('model_dropdown', 'value'),
        )
    def generate_text_output(btn_click, more_text_click, uncert_estimator, question_val, genAlg_val, model_val):
        
        show_more_style = {'display':'none'}
        if None in [uncert_estimator, question_val, genAlg_val, model_val]:
            return uqComponents.noResultsComp, html.H5('...', style={'line-height':'80px', 'color':'black'}), show_more_style
        
        ctx = callback_context
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        textResult, uncerEstResult = uqDataProcessor.generateTextOutput(model_val, genAlg_val ,question_val, uncert_estimator, data)
        
        if input_id == 'output_button' and len(textResult) > 5:
            textResult = textResult[:5]
            show_more_style = {'display':'block'}
        
        elif input_id == 'more_text_button':
            show_more_style = {'display':'none'}

        return textResult, uncerEstResult, show_more_style


    @app.callback(
        Output('output_button', 'disabled'),
        Input('uncert_est_dropdown', 'value'),    
        Input('question_dropdown', 'value'),
        Input('genAlg_dropdown', 'value'),
        Input('model_dropdown', 'value'),
        # prevent_initial_call=True
        )
    def update_go_button(uncert_estimator, question_val, genAlg_val, model_val):     
        if (uncert_estimator and question_val and genAlg_val and model_val): isDisabled = False
        else: isDisabled = True
        return isDisabled

   
