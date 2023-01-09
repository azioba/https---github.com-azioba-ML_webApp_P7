import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split

import streamlit as st 
import pickle
import requests
import shap
import json
import lime
import lime.lime_tabular

st.set_page_config(
    page_title="Modèle de Scoring",
    page_icon="🎈",
)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
         par *MAHAMADOU HAMIDOU Abdoul Aziz*
         """)

df = pd.read_csv('df_500.csv',index_col='SK_ID_CURR')
X = df.drop(columns=['TARGET'])
feature_names = X.columns

# Pickle
with open("classifier.pkl","rb") as pickle_in:
    classifier = pickle.load(pickle_in)
    
explainer = shap.TreeExplainer(classifier, X, feature_names=feature_names)
shap.initjs()

    
# explain model prediction shap results
def explain_model_prediction_shap(data):
    # Calculate Shap values
    shap_values = explainer(np.array(data))
    p = shap.plots.bar(shap_values)
    return p, shap_values 

# explain model prediction Lime results
def explain_model_prediction_lime(data, client):
    explaine_ = lime.lime_tabular.LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=['no_risk','risked'],
        mode = 'classification')
    
    instance = data.iloc[client]
    explanation = explainer_.explain_instance(
    instance,
    classifier.predict_proba,
    num_features=10,
    top_labels=1 )
    
    return explanation
    
    
# get the data of the selected customer
def get_value(index):
    # Select the row at the specified index
    value = X.loc[index]
    return value.values.tolist()
  
def request_prediction(URI, data):
    
    response = requests.post(URI, json=data)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    response = json.loads(response.text)
    response = pd.DataFrame(response)
    
    prediction = response['prediction'][0]
    probability = response['probability'][0]
    result = {'prediction':prediction, 'probability' : probability}
    return  result

def best_classification(probas, threshold, X):
    y_pred = 1 if probas > threshold else 0 
    return y_pred

def main():
    
    URI = 'http://127.0.0.1:5000/predict'
     
    st.title("Loan Default Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">loan payment risk prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Customer = st.sidebar.selectbox("Selectionner un numéro client: ",X.index)
    
    if st.sidebar.button("Predict"):
            data = get_value(Customer)
            result = request_prediction(URI, data)
            score = result['prediction']
            prob = result['probability']
            y_pred = best_classification(prob, 0.3918, data)
            if (y_pred == 1):
                risk_assessment = "Crédit refusé"
            else:
                risk_assessment = "Crédit accepté"
            st.sidebar.success(risk_assessment)
            st.sidebar.write("Prediction: ", int(score))
            st.sidebar.write("Probability: ", round(float(prob),4))
            st.sidebar.write("y_pred best threshold: ", y_pred)
            st.subheader('Result Interpretability - Applicant Level')
            p, shap_values = explain_model_prediction_shap(data) 
            st.pyplot(p)
            st.subheader('Model Interpretability - Overall') 
            shap_values_ttl = explainer(X) 
            fig_ttl = shap.plots.bar(shap_values_ttl, max_display=10)
            st.pyplot(fig_ttl)
            st.pyplot(shap.summary_plot(shap.TreeExplainer(classifier).shap_values((X)), X, plot_type="bar"))
            
            
            explanation = explain_model_prediction_lime(df, Customer)
            explanation.show_in_notebook(show_all=False)
    
    
    select = st.sidebar.checkbox('Features')
    if select:
            selected_feature = st.sidebar.selectbox('Selectionner une feature', feature_names)
            if st.sidebar.button('display'):
                st.bar_chart(df.groupby("TARGET").selected_feature.value_counts())
            
if __name__=='__main__':
    main() 