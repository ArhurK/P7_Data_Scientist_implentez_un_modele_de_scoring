####### Pour tester API en local ########
# uvicorn nom_de_votre_module:app --reload
# uvicorn vs_code_api:app --reload
###########################################


# Librairies
import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI , HTTPException
import json
import numpy as np
import shap
# from flask import Flask, request, jsonify
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse



# Charger le modèle MLFlow
model_path = r"C:\Users\Hankour\OneDrive\Bureau\OC_Arthur\mlruns\159852288404653738\89e2dedfd3dd428b849adecd8c60de14\artifacts\model_lgbm_class_weight_best_model"
model = mlflow.sklearn.load_model(model_path)
df = pd.read_pickle('test_df.pkl')

# débug
import logging
logging.basicConfig(level=logging.DEBUG)  # Active les logs détaillés

# Initialisation
app = FastAPI(debug=True)

origins = [
    "http://localhost",
    "http://10.137.42.118:8501"  #  Streamlit sur port local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# api test
@app.get('/')
def Hello():
    return {'Hello ceci est un test' : 'test 1'}

# api qui obtient le predict proba du modèle
@app.get('/predict_proba') 
def predict_proba(id_client : dict):

    # Sélection ID client
    print(id_client)
    try:
        selected_row = pd.DataFrame(df.loc[df['SK_ID_CURR'] == int(id_client['index'])])
    except IndexError:
        print('ID Not found')
    else:
        # Acces à la ligne sélectionnée
        print(selected_row)
    
    # Sélection des features
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = selected_row[feats]
    X = pd.DataFrame(X)

    # Effectuer la prédiction
    predictions = model.predict_proba(X)
    print(predictions)

    # Convertir les prédictions en format JSON
    # Faire plutot f('prediction{id_client}')
    result = {'predictions': predictions.tolist()}

    return result

################################
## SHAP VALUES #############
##############################


@app.get('/shap')
def shap_vector(id_client: dict):
    try:
        selected_row = df.loc[df['SK_ID_CURR'] == int(id_client['index'])]
        if selected_row.empty:
            raise HTTPException(status_code=404, detail='ID Not found')

        feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        if not all(col in selected_row.columns for col in feats):
            raise HTTPException(status_code=404, detail='Column Not found')

        X = selected_row[feats]

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)
        if shap_values is None or len(shap_values) != 2:
            raise HTTPException(status_code=500, detail='Error in Shapley values calculation')

        shap_values_list = shap_values[1].tolist()

        shap_values_dict = {
            'shap_values': shap_values_list,
            'X': X.to_dict(),
        }

        shap_values_dict_json = json.dumps(shap_values_dict)

        return shap_values_dict_json

    except IndexError:
        raise HTTPException(status_code=404, detail='ID Not found')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))















# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

# @app.get('/shap')
# def shap_vector(id_client: dict):
#     try:
#         selected_row = df.loc[df['SK_ID_CURR'] == int(id_client['index'])]
#     except IndexError:
#         print('ID Not found')
#         return HTTPException(status_code=404, detail='ID Not found')
#     else:
#         # Access the selected row
#         print(selected_row)

#     # Select features
#     feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
#     X = selected_row[feats]

#     # Make predictions
#     predictions = model.predict_proba(X)

#     # Calculate Shapley values
#     explainer = shap.TreeExplainer(model)

#     # Conservant le format DataFrame
#     shap_values = explainer.shap_values(X)

#     # Serialize Shapley values using the custom encoder
#     # shap_values_json = json.dumps({'shap_values': shap_values}, cls=NumpyEncoder)

#     shap_values_list = shap_values[1].tolist()

#     # Créez un dictionnaire pour stocker les Shapley values
#     shap_values_dict = {}

#     key = 'shap_values'
#     key_X = 'X'
#     key_X_columns ='X_columns'

#     shap_values_dict[key] = shap_values_list
#     shap_values_dict[key_X] = X.to_dict()
#     # shap_values_dict[key_X_columns] = list(X.columns)


#     shap_values_dict_json = json.dumps(shap_values_dict)

#     return shap_values_dict_json



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
