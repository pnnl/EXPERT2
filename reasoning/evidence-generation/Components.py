
from dash import html
import dash_bootstrap_components as dbc

class CompDefaultSelGen():
    def __init__(self, graph_data, language_data, graph_text_link):
        self.graph_data_df = graph_data
        self.lang_df = language_data
        self.graph_text_link_df = graph_text_link
        self.primary_dropdown_options = CompDataLoader.load_dropdown_options(self.lang_df['prompt'].to_list(), self.lang_df.text_query_id.to_list())
    


class CompDataLoader:
    @staticmethod
    def load_dropdown_options(labels, values = None, comp_type = None):
        if values: options = [{'label':dd_lab, 'value':dd_val} for dd_lab, dd_val in zip(labels, values)]
        else: options =[{'label':dd_val, 'value':dd_val} for dd_val in labels]
        return options

    @staticmethod
    def get_query_outputs(answers):
        output_ans_comp = []
        for ans in answers:
            ans_comp = dbc.Card([ dbc.CardHeader(html.P(ans['text'], style ={'margin':'0px'}), style={'padding-top':'2px', 
                                                                            'text-wrap':'wrap', 'word-wrap':'break-word','margin':'0px'}), 
                                  dbc.CardBody([html.P('Uncertainty:', style={ 'color':'#635f71','font-size':'0.9rem', 
                                  'padding':'0px', 'margin':'0px', 'margin-right':'0.3rem'}),
                                                html.P(str(float("{0:.4f}".format(ans['confidence']))*100)[:5]+"%", 
                                                style = {'font-weight':'bold', 'font-size':'0.9rem','padding':'0px', 'margin':'0px'})], 
                                                style={'padding-top':'0.2rem', 'display':'flex', 'flex-wrap':'wrap'})
                               ], style={'min-height':'40%', 'max-height':'52%'})
            output_ans_comp.append(ans_comp)
        return output_ans_comp



    @staticmethod
    def get_modal_output(df, segment_id):
        
        full_text = df.query('segment_id==@segment_id')['full_text'].to_list()[0]
        text_segment = df.query('segment_id==@segment_id')['text_segment'].to_list()[0]
        text_chunks = full_text.split(text_segment)
        title = html.P(df.query('segment_id==@segment_id')['title'].to_list()[0])
        body = html.Div([html.P(text_chunks[0]),
                        html.P(text_segment, style = {'background-color':'yellow'}),
                        html.P(text_chunks[1])], style = {'text-align':'justify'})
        return title, body

    @staticmethod
    def get_source_doc_outputs(df, rsn_see_all, rsn_sort_val):
        # Sort by date
        # Take top 6 
        if not rsn_see_all: df = df.head(6)
        else: df = df.head(10)
        if rsn_sort_val == 'most_recent': df.sort_values(by='publication_date', ascending = False , inplace = True)
        else: df.sort_values(by='publication_date', inplace = True )
        output_ans_comp = []
        title_list = df.title.to_list()
        text_list = df.full_text.to_list()
        text_seg_list  = df.text_segment.to_list()
        pub_date_list = df.publication_date.to_list()
        segment_id_list = df.segment_id.to_list()

        #Meta Data
        author_list = df.author.to_list()
        venue_list = df.pub_venue.to_list()
        topic_list = df.topic.to_list()

        i = 0
        
        for title, text, date, segment, seg_id, author, venue, topic in zip(title_list, text_list, pub_date_list, text_seg_list, segment_id_list, 
            author_list, venue_list, topic_list):
            # text_chunks = text.split(segment)
            ans_comp = dbc.Card([  
                                  dbc.CardBody([
                                                
                                                dbc.Badge(date, pill=True, color="secondary", className="me-1", 
                                                          style ={'float':'left', 'margin-top':'-12px'}),
                                                html.Div([html.I(id = f"segment_info_{i}", className="fa fa-info-circle fa-sm", 
                                                            style={'display':'flex', 
                                                                    'color':'grey',
                                                                   'align-items':'center', 
                                                                   'font-size':'1rem', 
                                                                   }
                                                            ), 

                                                        dbc.Button(id = {"type":'expand_button',"index":seg_id}, className="fa fa-expand fa-sm", 
                                                            style={'display':'flex', 
                                                                    'color':'grey',
                                                                   'align-items':'center', 
                                                                   'font-size':'1rem', 
                                                                   'background-image':'none',
                                                                   'background-color':'rgba(0,0,0,0)',
                                                                   'border':'none'
                                                                  }
                                                            ), 


                                                ], style={'display':'flex', 'justifyContent':'flex-end'}),

                                                 dbc.Tooltip([html.P(f"Venue: {venue}", 
                                                                    style={'padding':'0px', 'margin':'0px', 'font-weight':'525', 'text-align':'left',
                                                                    'margin-bottom':'0.5rem'}
                                                                    ),
                                                             html.P(f"Author(s): {author}",
                                                                   style={'padding':'0px', 'margin':'0px', 'font-weight':'525', 'text-align':'left','margin-bottom':'0.5rem'}),
                                                             html.P(f"Topic: {topic}", 
                                                                    style={'padding':'0px', 'margin':'0px', 'font-weight':'525','text-align':'left'}
                                                                    ), ], 
                                                            
                                                             target = f"segment_info_{i}", 
                                                             placement = 'right', 
                                                             style={'background-color':'white', 
                                                                    'color':'black',
                                                                    'font-size':'0.75rem',
                                                                    'text-align':'left',
                                                                    'border':'1px solid black', 'padding':'5px', 
                                                                    'margin':'0px'}
                                                                    ), 


                                                html.P(title, style={'margin-right':'0.3rem', 'color':'#635f71','margin-top':'16px',
                                                                            'font-size':'0.9rem', 'font-weight':'bold', 
                                                                               "line-height": "1.5em",
                                                                                "height": "3em",
                                                                                "overflow": "hidden",
                                                                                "text-overflow": "ellipsis",
                                                                                "width": "100%"}),
                                                html.Div([html.P("..."),#text_chunks[0]
                                                          html.P(segment),
                                                          html.P("...") #text_chunks[0]
                                                         
                                                        ], 
                                                        style = {'height':'178px', 'font-size':'0.9rem', 'padding':'0px',
                                                'overflowY':'scroll', 'background-color':'#f7f7f7', 'border-radius':'5px'})
                                                
                                                
                                                ], 
                                                style={'padding':'2px'})
                               ], style={'height':'285px', 'margin-top':'15px'}, className = 'text-card',)
            output_ans_comp.append(ans_comp)
            i+=1
        return output_ans_comp


