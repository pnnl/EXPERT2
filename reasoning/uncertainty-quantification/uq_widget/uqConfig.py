# import uqColors, uqWidget
from . import uqColors, uqWidget
import pandas as pd
import os 
import numpy as np 

def replaceText(textList):
    for text in textList:
        text[0] = text[0].replace('\n', "\\n")
        text[1] = [t.replace('\n', '\\n') for t in text[1]]
    return textList

def getTokenList(textList):
    tokenList = [text[0] for text in textList]
    return tokenList

def getHoverList(textList):
    hoverList = []
    for text in textList:
        tmp = ''
        for tok, prob in zip(text[1], text[2]):
            tmp+=str(tok)+" : "+str(round(float(prob), 3))+"\n"
        hoverList.append(tmp[:-2])
     

    return hoverList

def getProbabilityList(textList):
    probabilityList = []
    for text in textList:
        probVal = -1
        if text[0] in text[1]: probVal = text[1].index(text[0]) #probVal = text[2][text[1].index(text[0])]
        probabilityList.append(probVal)                                               
    return probabilityList

def getColorList(probList):
    
    probList = pd.DataFrame(probList).rename(columns={0:'Prob_List'})
    probList['Prob_List'] = probList['Prob_List'].astype(float)

#     conditions =[((probList['Prob_List'] > 0) & (probList['Prob_List'] <= 0.1)),
#             ((probList['Prob_List'] > 0.1) & (probList['Prob_List'] <= 0.2)),
#             ((probList['Prob_List'] > 0.2) & (probList['Prob_List'] <= 0.3)),
#             ((probList['Prob_List'] > 0.3) & (probList['Prob_List'] <= 0.4)),
#             ((probList['Prob_List'] > 0.4) & (probList['Prob_List'] <= 0.5)),
#             ((probList['Prob_List'] > 0.5) & (probList['Prob_List'] <= 0.6)),
#             ((probList['Prob_List'] > 0.6) & (probList['Prob_List'] <= 0.7)),
#             ((probList['Prob_List'] > 0.7) & (probList['Prob_List'] <= 0.8)),
#             ((probList['Prob_List'] > 0.8) & (probList['Prob_List'] <= 0.9)),
#             ((probList['Prob_List'] > 0.9) & (probList['Prob_List'] <= 1)),
#             (probList['Prob_List'] == -1)
#             ]

    conditions =[(probList['Prob_List'] == 0),(probList['Prob_List'] == 1),(probList['Prob_List'] == 2),
                 (probList['Prob_List'] == 3),(probList['Prob_List'] == 4),(probList['Prob_List'] == 5),
                 (probList['Prob_List'] == 6),(probList['Prob_List'] == 7),(probList['Prob_List'] == 8),
                 (probList['Prob_List'] == 9),(probList['Prob_List'] == -1)
            ]
    values = uqColors.text_colors_rgba
    colorList = np.select(conditions, values)

    return colorList


def generateDictInputs(inputList, key):
    if type(key) == list: values = key
    else: values = [f"{key}_{i}" for i in range(len(inputList))]    

    dictResult = dict(zip(values, inputList))
    optionsResult = [{'label':inputVal , 'value':value} for inputVal, value in zip(inputList, values)]
    return dictResult, optionsResult


def processData(filepath):
    jsonFiles = [file for file in os.listdir(filepath) if '.json' in file]



    df_raw = pd.DataFrame()
    for file in jsonFiles:
        df = pd.read_json(os.path.join(filepath, file)).reset_index().rename(columns={'index':'Text'})
        # Process the text to display escape sequences
        df['Text'] = df['Text'].str.replace('\n','\\n')
        df['Generation'] = df['Generation'].apply(lambda x: replaceText(x))
        df['Token_List'] = df['Generation'].apply(lambda x: getTokenList(x))
        df['Hover_List'] = df['Generation'].apply(lambda x: getHoverList(x))
        df['Probability_List'] = df['Generation'].apply(lambda x: getProbabilityList(x))
        df['Color_List'] = df['Probability_List'].apply(lambda x: getColorList(x))
    #     df['Input_Question'] = inputQuestion
        
        df_raw = df_raw.append(df).reset_index(drop=True)
        
    questionList = list(df_raw['Input_Question'].unique())
    modelList = list(df_raw['Model'].unique())
    genAlgoList = list(df_raw['Gen_Algo'].unique())

    df_raw = df_raw.groupby(['Model', 'Gen_Algo', 'Input_Question'])


    uncertainityEstimatorList = ['Normalized Entropy', 'Entropy','Lexical Similarity', 'Semantic Uncertainty']
    uncertainityEstimatorDict, uncertainityEstimatorOptions = generateDictInputs(uncertainityEstimatorList, ['Normalized_Entropy', 'Entropy', 'Lexical_Similarity', 'Semantic Uncertainty'])

    questionDict, questionOptions = generateDictInputs(questionList, 'question')
    modelDict, modelOptions = generateDictInputs(modelList, 'model')
    genAlgoDict, genAlgoOptions = generateDictInputs(genAlgoList, 'genAlgo')
    
    outputDict = {'df_raw':df_raw,
                    'uncertainityEstimatorOptions':uncertainityEstimatorOptions,
                    'questionDict':questionDict,  
                    'questionOptions':questionOptions,
                    'modelDict':modelDict, 
                    'modelOptions':modelOptions,
                    'genAlgoDict':genAlgoDict, 
                    'genAlgoOptions': genAlgoOptions
                    }

    return outputDict

# # Temperature Settings
# temperatureRange = [0.5, 1]
# temperatureStep = 0.5