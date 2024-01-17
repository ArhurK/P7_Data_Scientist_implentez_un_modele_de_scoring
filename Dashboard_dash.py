# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import requests
# import json
# import shap
# import numpy as np
# import pandas as pd
# import logging

# app = dash.Dash(__name__)

# # Layout of the Dash app
# app.layout = html.Div([

#     html.H1("Dashboard modèle prédictif risque de défault"),
    
#     # Entrée pour l'ID client
#     dcc.Input(id='client-id', type='text', value='403414'),
    
#     # Bouton pour déclencher l'envoi de la requête
#     html.Button('Obtenir les prédictions et Shap values', id='button'),
    
#     # Affichage des prédictions
#     html.Div(id='prediction-output'),
    
#     # Graphique pour les Shap values
#     dcc.Graph(id='shap-plot')
# ])

# # Callback pour mettre à jour les prédictions et le graphique Shap values
# @app.callback(
#     [Output('prediction-output', 'children'),
#      Output('shap-plot', 'figure')],
#     [Input('button', 'n_clicks')],
#     [dash.dependencies.State('client-id', 'value')]
# )

# # def update_output(n_clicks, client_id):
# #     if n_clicks is None:
# #         # Aucun clic sur le bouton, aucune action nécessaire
# #         return dash.no_update, dash.no_update

# def update_output(n_clicks, client_id):
#     logging.info(f"Button clicked for client ID: {client_id}")

#     if n_clicks is None:
#         # Aucun clic sur le bouton, aucune action nécessaire
#         return dash.no_update, dash.no_update
    
#     # API Prédictions
#     predict_proba_url = requests.get(url='http://127.0.0.1:8000/predict_proba', json = {"index": client_id })
#     response_predict_proba = predict_proba_url

#     # Appel API pour obtenir les Shap values
#     shap_url = requests.get(url='http://127.0.0.1:8000/shap', json = {"index": client_id })
#     response_shap = shap_url
    
#     if response_predict_proba.status_code == 200 and response_shap.status_code == 200:

#         # Récupérer les données

#         # Predict_proba
#         data_predict_proba = response_predict_proba.json()
#         # predictions = data_predict_proba['predictions']
#         value = data_predict_proba['predictions'][0][1]
#         # prediction_value = percent_func(value)
#         prediction_value = value

#         # Shap values
#         data_shap = response_shap.json()
#         data_shap = json.loads(data_shap.text)
#         shap_json_dict = json.loads(data_shap)
#         shap_values = shap_json_dict["shap_values"]

#         X_shap = shap_json_dict['X']
#         X_shap = pd.DataFrame(X_shap)


#         # Affichage des prédictions
#         prediction_output = (f"Probabilité de faire défault : {prediction_value} %")
#         # Création d'un graphique Shap values
#         shap_plot = shap.summary_plot(shap_values, X_shap , feature_names= X_shap.columns)

#         return prediction_output, shap_plot
#     else:
#         return "Erreur lors de la récupération des données", {}

# if __name__ == '__main__':
#     app.run_server(debug=True)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import json
import shap
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Dashboard modèle prédictif risque de défault"),
    dcc.Input(id='client-id', type='text', value='403414'),
    html.Button('Obtenir les prédictions et Shap values', id='button'),
    html.Div(id='prediction-output'),
    dcc.Graph(id='shap-plot')
])

def update_output(n_clicks, client_id):
    if n_clicks is None:
        # No button click, no action needed
        return dash.no_update, dash.no_update

    try:
        # API Prédictions
        predict_proba_url = requests.get(url='http://127.0.0.1:8000/predict_proba', json={"index": client_id})
        response_predict_proba = predict_proba_url

        # Appel API pour obtenir les Shap values
        shap_url = requests.get(url='http://127.0.0.1:8000/shap', json={"index": client_id})
        response_shap = shap_url

        if response_predict_proba.status_code == 200 and response_shap.status_code == 200:
            # Récupérer les données
            data_predict_proba = response_predict_proba.json()
            value = data_predict_proba['predictions'][0][1]
            prediction_value = value

            data_shap = response_shap.json()
            data_shap = json.loads(data_shap.text)
            shap_json_dict = json.loads(data_shap)
            shap_values = shap_json_dict["shap_values"]

            X_shap = shap_json_dict['X']
            X_shap = pd.DataFrame(X_shap)

            # Affichage des prédictions
            prediction_output = (f"Probabilité de faire défault : {prediction_value} %")
            
            # Création d'un graphique Shap values
            shap_plot = shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns)

            return prediction_output, shap_plot
        else:
            print(f"API request failed. Predict Proba status: {response_predict_proba.status_code}, Shap status: {response_shap.status_code}")
            return f"Erreur lors de la récupération des données. Predict Proba status: {response_predict_proba.status_code}, Shap status: {response_shap.status_code}", {}

    except Exception as e:
        print(f"Exception during API request: {str(e)}")
        return f"Erreur inattendue: {str(e)}", {}

if __name__ == '__main__':
    app.run_server(debug=True)
