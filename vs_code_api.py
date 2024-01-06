####### Pour tester API en local
# uvicorn nom_de_votre_module:app --reload


# Librairies
import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI
import json
import numpy as np
import shap
from flask import Flask, request, jsonify


# Charger le modèle MLFlow
model_path = r"C:\Users\Hankour\OneDrive\Bureau\OC_Arthur\mlruns\159852288404653738\89e2dedfd3dd428b849adecd8c60de14\artifacts\model_lgbm_class_weight_best_model"
model = mlflow.sklearn.load_model(model_path)
df = pd.read_pickle('test_df.pkl')

# Initialisation
app = FastAPI()

# api test
@app.get('/')
def Hello():
    return {'Hello ceci est un test' : 'test 1'}

# api qui obtient le predict proba du modèle
@app.get('/predict_proba') # get pas post
def predict_proba(id_client : dict):
    
    # Charger les données du formulaire
    #     data = request.json
    #     df = pd.DataFrame(data)
    # df_test = pd.read_pickle('test_df.pkl')
    
    # Sélection ID client
    print(id_client)
    try:
        selected_row = df.loc[df['SK_ID_CURR'] == id_client['index']]
    except IndexError:
        print('ID Not found')
    else:
        # Acces à la ligne sélectionnée
        print(selected_row)
    
    # Sélection des features
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = selected_row[feats]

    # Effectuer la prédiction
    predictions = model.predict_proba(X)
    print(predictions)

    # Convertir les prédictions en format JSON
    # Faire plutot f('prediction{id_client}')
    result = {'predictions': predictions.tolist()}

    return result


################################


# # Load your DataFrame and model
# df = pd.read_csv('your_dataframe.csv')  # Replace with your actual DataFrame
# model = your_loaded_model  # Replace with your actual model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.get('/shap')
def shap_vector(id_client: dict):
    try:
        selected_row = df.loc[df['SK_ID_CURR'] == id_client['index']]
    except IndexError:
        print('ID Not found')
    else:
        # Access the selected row
        print(selected_row)

    # Select features
    feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X = selected_row[feats]

    # Make predictions
    predictions = model.predict_proba(X)

    # Calculate Shapley values
    explainer = shap.TreeExplainer(model)

    # Convert the DataFrame X to a NumPy array
    # X_array = X.values
    # shap_values = explainer.shap_values(X_array)

    # Conservant le format DataFrame
    shap_values = explainer.shap_values(X)

    # Serialize Shapley values using the custom encoder
    shap_values_json = json.dumps({'shap_values': shap_values}, cls=NumpyEncoder)

    return shap_values_json



# @app.get('/shap')
# def shap_vector(id_client : dict):

#     try:
#         selected_row = df.loc[df['SK_ID_CURR'] == id_client['index']]
#     except IndexError:
#         print('ID Not found')
#     else:
#         # Acces à la ligne sélectionnée
#         print(selected_row)
    
#     # Sélection des features
#     feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
#     X = selected_row[feats]

#     # Effectuer la prédiction
#     predictions = model.predict_proba(X)

#     # Calculer les Shapley values
#     explainer = shap.TreeExplainer(model)
#     # Convertir le DataFrame X en une matrice NumPy
#     X_array = X.values
#     shap_values = explainer.shap_values(X_array)
#     shap_values_dict = {'shap_values' : shap_values}
#     return shap_values_dict
