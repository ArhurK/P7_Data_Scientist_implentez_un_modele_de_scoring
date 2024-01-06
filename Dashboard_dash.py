import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import json

app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([

    html.H1("Dashboard pour le modèle"),
    
    # Entrée pour l'ID client
    dcc.Input(id='client-id', type='text', value='403414'),
    
    # Bouton pour déclencher l'envoi de la requête
    html.Button('Obtenir les prédictions et Shap values', id='button'),
    
    # Affichage des prédictions
    html.Div(id='prediction-output'),
    
    # Graphique pour les Shap values
    dcc.Graph(id='shap-plot')
])

# Callback pour mettre à jour les prédictions et le graphique Shap values
@app.callback(
    [Output('prediction-output', 'children'),
     Output('shap-plot', 'figure')],
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('client-id', 'value')]
)
def update_output(n_clicks, client_id):
    # Appel API pour obtenir les prédictions
    # predict_proba_url = f"http://127.0.0.1:8000/predict_proba?index={client_id}"
    # response_predict_proba = requests.get(predict_proba_url)

    predict_proba_url = requests.get(url='http://127.0.0.1:8000/predict_proba', json = {"index": client_id })
    response_predict_proba = predict_proba_url

    # Appel API pour obtenir les Shap values
    # shap_url = f"http://127.0.0.1:8000/shap?index={client_id}"
    # response_shap = requests.get(shap_url)

    shap_url = requests.get(url='http://127.0.0.1:8000/shap', json = {"index": client_id })
    response_shap = shap_url
    
    if response_predict_proba.status_code == 200 and response_shap.status_code == 200:
        # Récupérer les données
        data_predict_proba = response_predict_proba.json()
        predictions = data_predict_proba['predictions']

        data_shap = response_shap.json()
        shap_values = data_shap['shap_values']

        # Affichage des prédictions
        prediction_output = f"Predictions: {predictions}"

        # Création d'un graphique Shap values
        # (Remplacez cela par votre propre logique de création de graphique)
        shap_plot = {
            'data': [{'x': list(range(len(shap_values))), 'y': shap_values, 'type': 'bar', 'name': 'Shap Values'}],
            'layout': {'title': 'Shap Values'}
        }

        return prediction_output, shap_plot
    else:
        return "Erreur lors de la récupération des données", {}

if __name__ == '__main__':
    app.run_server(debug=True)

# import requests
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import requests
# import json
# import mlflow.sklearn
# import pandas as pd

# app = dash.Dash(__name__)

# # Layout of the Dash app
# app.layout = html.Div([

#     html.H1("Dashboard pour le modèle"),
    
#     # Entrée pour l'ID client
#     dcc.Input(id='client-id', type='text', value='403414'),
    
#     # Bouton pour déclencher l'envoi de la requête
#     html.Button('Obtenir les prédictions et Shap values', id='button'),
    
#     # Affichage des prédictions
#     html.Div(id='prediction-output'),
    
#     # Graphique pour les Shap values
#     dcc.Graph(id='shap-plot')
# ])

# @app.callback(
#     [Output('prediction-output', 'children'),
#      Output('shap-plot', 'figure')],
#     [Input('button', 'n_clicks')],
#     [dash.dependencies.State('client-id', 'value')]
# )
# def update_output(n_clicks, client_id):
#     if n_clicks is None:
#         # Aucun clic sur le bouton, aucune action nécessaire
#         return dash.no_update, dash.no_update
    
#     # Appel API pour obtenir les prédictions
#     predict_proba_url = 'http://127.0.0.1:8000/predict_proba'
#     response_predict_proba = requests.get(predict_proba_url, json={"index": client_id})

#     # Appel API pour obtenir les Shap values
#     shap_url = 'http://127.0.0.1:8000/shap'
#     response_shap = requests.get(shap_url, json={"index": client_id})

#     if response_predict_proba.status_code == 200 and response_shap.status_code == 200:
#         # Récupérer les données
#         data_predict_proba = response_predict_proba.json()
#         predictions = data_predict_proba['predictions']

#         data_shap = response_shap.json()
#         shap_values = data_shap['shap_values']

#         # Affichage des prédictions
#         prediction_output = f"Predictions: {predictions}"

#         # Création d'un graphique Shap values
#         shap_plot = {
#             'data': [{'x': list(range(len(shap_values))), 'y': shap_values, 'type': 'bar', 'name': 'Shap Values'}],
#             'layout': {'title': 'Shap Values'}
#         }

#         return prediction_output, shap_plot
#     else:
#         print(f"Erreur lors de la récupération des données - Predict Proba: {response_predict_proba.content}")
#         print(f"Erreur lors de la récupération des données - Shap: {response_shap.content}")
#         return "Erreur lors de la récupération des données", {}
