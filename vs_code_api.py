import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI

# Charger le modèle MLFlow
model_path = r"C:\Users\Hankour\OneDrive\Bureau\OC_Arthur\mlruns\159852288404653738\89e2dedfd3dd428b849adecd8c60de14\artifacts\model_lgbm_class_weight_best_model"
model = mlflow.sklearn.load_model(model_path)
df = pd.read_pickle('test_df.pkl')

from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn
import shap

app = FastAPI()

@app.get('/')
def Hello():
    return {'Hello ceci est un test' : 'test 1'}

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

@app.get('/shap')
def shap_vector(id_client : dict):

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

    # Calculer les Shapley values
    explainer = shap.TreeExplainer(model)
    # Convertir le DataFrame X en une matrice NumPy
    X_array = X.values
    shap_values = explainer.shap_values(X_array)
    shap_values_dict = {'shap_values' : shap_values}
    return shap_values_dict
