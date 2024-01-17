import streamlit as st
import requests
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI 

# Titre du Dashboard
st.title("Dashboard risque de défaut")

# Interface utilisateur Streamlit
client_id = st.text_input("Renseigner l'ID du client (6 chiffres), puis cliquer sur le bouton 'Obtenir les prédictions et Shap values'", '403414')
button_clicked = st.button('Obtenir les prédictions et Shap values')

if button_clicked:
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
            prediction_value = data_predict_proba['predictions'][0][1]

            data_shap = response_shap.json()
            shap_json_dict = json.loads(data_shap) 
            print(f'shap_json_dict : {type(shap_json_dict)}')

            shap_values = shap_json_dict["shap_values"]
            print(f'shap_values : {type(shap_values)}')

            shap_values_array = np.array(shap_values)
            print(f'shap_values_array : {type(shap_values_array)}')

            X_shap = shap_json_dict['X']
            X_shap = pd.DataFrame(X_shap)

            # Affichage des prédictions
            st.subheader(f"Risque de défault du client n {client_id}")
            prediction_percentage = round(prediction_value * 100, 1)
            # st.write(f"Probabilité de défaut du client: {prediction_percentage}%")

            # # Afficher le compteur plot
            # st.subheader(f"Risque de défaut du client n {client_id}")
            # # Calcul de la probabilité prédite
            # prediction_percentage = round(prediction_value * 100, 1)
            # Définir la couleur en fonction de la probabilité
            progress_color = 'green' if prediction_percentage <= 50 else 'red'
            # Afficher le compteur plot avec la couleur appropriée
            # st.progress(prediction_percentage / 100).progress_style(progress_color)
            # Afficher la probabilité
            st.markdown(f'<p style="color:{progress_color}; font-size:30px;">Probabilité de défaut du client: {prediction_percentage}%</p>', unsafe_allow_html=True)



            # Affichage des données du client
            st.subheader("Données du client")
            st.dataframe(X_shap)


            # Création d'un graphique Shap values avec Matplotlib
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_array, X_shap, feature_names=X_shap.columns, show=False)
            plt.tight_layout()

            # Afficher le graphique avec Streamlit
            st.subheader("Shap-values : Importance des variables prédictives")
            st.pyplot(fig)
            
        else:
            st.error(f"Erreur lors de la récupération des données. Predict Proba status: {response_predict_proba.status_code}, Shap status: {response_shap.status_code}")

    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")



