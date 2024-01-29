# tests/test_main.py
import json
import pytest
from fastapi.testclient import TestClient
from ..vs_code_api import predict_proba, shap_vector, app


def test_predict_proba():
     id_client = {'index' : 403414}
     dict_res = predict_proba(id_client)
     res_proba = dict_res['predictions'][0][1]
  
     assert type(dict_res) == dict
     assert type(res_proba) == float
     assert (0 < res_proba < 1 )

def test_shap_vector():
     id_client = {'index' : 403414}
     shap_values_json = shap_vector(id_client)
     shap_values_dict = json.loads(shap_values_json)
     shap_list = shap_values_dict['shap_values']
     X_dict = shap_values_dict['X']

     assert type(shap_values_json) == str
     assert type(shap_values_dict) == dict
     assert type(shap_list) == list
     assert type(X_dict) == dict
