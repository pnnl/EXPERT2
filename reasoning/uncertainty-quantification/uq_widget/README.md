**Uncertainty Quantification Widget**

The widget provides the user interface to demonstrate the capabilities of uncertainty quantification pipeline. To run the widget, install the dependencies using the command "pip install 'uqBeta-0.1-py3-none-any.whl' --force-reinstall". Run the cell under "Demo" in the notebook to load the widget. 

Once the widget is loaded and the inputs (Generator Algorithm, Uncertainty Estimator and the Input Question) are selected, the widget displays series of answers to the input question with colored background corresponding to the rank in which the token is displayed. 

Hover on each token to view more related tokens with their corresponding probabilities. The widget also displays the uncertainty estimation measured in terms of Entropy, Normalized Entropy, Lexical Similarity and Semantic Uncertainty. 
    