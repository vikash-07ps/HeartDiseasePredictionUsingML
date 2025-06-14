import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Prediction CSV</a>'
    return href

st.markdown("<h1 style='color:red;'>Heart Disease Predictor</h1>",unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(['Predict','Bulk Predict', 'Model Information'])


st.sidebar.title("About the Data")
st.sidebar.markdown("""
        1. No NaN values allowed.
        2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
        3. Feature values conventions:\n
            - Age: age of the patient [Years]
            - Sex: sex of the patient [0: Male, 1: Female]\n
            - ChestPainType: chest pain type [0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymtomatic, 3: Typical Angina]\n
            - RestingBP: resting Blood pressure [mm Hg]\n
            - Cholesterol: serum cholesterol [mm/dl]\n
            - FastingBS: fasting blood sugar [1: FastingBS > 120mg/dl, 0: otherwise]\n
            - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality, 2: Left Ventricular Hypertrophy]\n
            - MaxHR: maximum heart rate achieved [numeric value between 60 and 202]\n
            - ExerciseAngina: exercise-induced Angina [0: No, 1: Yes]\n
            - Oldpeak: oldpeak = ST [numeric value measured in depression]\n
            - ST_Slope: the slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]\n
            """)

with tab1:
     age=st.number_input("Age (Years)", min_value=0, max_value=150)
     sex=st.selectbox("Sex",["Male", "Female"])
     chest_pain=st.selectbox("Chest Pain Type",["Atypical Angina","Non-Anginal Pain","Asymptomatic"," Typical Angina"])
     resting_bp=st.number_input("Resting Blood Pressure (mm/dl)", min_value=0, max_value=300)
     cholesterol=st.number_input("Serum Cholestrol (mm/dl)", min_value=0)
     fasting_bs=st.selectbox("Fasting Blood Sugar", ["Less Than or Equal to 120 mg/dl", "Greater than 120 mg/dl"])
     resting_ecg=st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
     max_hr=st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
     exercise_angina=st.selectbox("Exercise-Enduced Angina", ["No","Yes"])
     oldpeak= st.number_input("Oldpeak (ST Depression)", min_value=0.0,max_value=10.0)
     st_slope=st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
     
     # Convert Categorical inputs to numeric
     sex=0 if sex=="Male" else 1
     chest_pain=["Atypical Angina","Non-Anginal Pain","Asymptomatic"," Typical Angina"].index(chest_pain)
     fasting_bs=1 if fasting_bs== "Greater than 120 mg/dl" else 0
     resting_ecg=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
     exercise_angina=1 if exercise_angina=="Yes" else 0
     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
     
     input_data = pd.DataFrame({
         'Age': [age],
         'Sex': [sex],
         'ChestPainType': [chest_pain],
         'RestingBP': [resting_bp],
         'Cholesterol': [cholesterol],
         'FastingBS': [fasting_bs],
         'RestingECG': [resting_ecg],
         'MaxHR': [max_hr],
         'ExerciseAngina': [exercise_angina],
         'Oldpeak': [oldpeak],
         'ST_Slope': [st_slope]
     })
     
     
     algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
     modelnames = ['Dtree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl']
     
     predictions=[]
     def predict_heart_disease(data):
         for modelname in modelnames:
             model= pickle.load(open(modelname,'rb'))
             prediction = model.predict(data)
             predictions.append(prediction)
         return predictions
     
     # Create a submit button to make predictions
     if "result" not in st.session_state:
        st.session_state.result = None

     if "predictions" not in st.session_state:
         st.session_state.predictions = None

     if "submitted" not in st.session_state:
         st.session_state.submitted = False
 
     if st.button("Submit"):
         st.session_state.result = predict_heart_disease(input_data)
         st.session_state.predictions = predictions  # Make sure 'predictions' is defined before this
         st.session_state.submitted = True

     if st.session_state.submitted and st.session_state.result is not None:
         st.subheader('Final Predicted Result')
        
         count = 0
         for i in range(len(st.session_state.predictions)):
             if st.session_state.result[i][0] == 1:
                 count += 1
        
         percentage = (count / len(st.session_state.predictions)) * 100
         color = "red" if percentage >= 50 else "green"
         st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:24px;'>This Patient has {percentage:.2f}% Chance of Heart Disease.</span>", unsafe_allow_html=True)

         st.markdown('************************************')

         if st.button("Detailed Result"):
             st.subheader("Results....")
             st.markdown('************************************')
 
             for i in range(len(st.session_state.predictions)):
                 st.subheader(algonames[i])
                 if st.session_state.result[i][0] == 0:
                     st.write("No Heart Disease Detected.")
                 else:
                     st.write("Heart Disease Detected.")
                 st.markdown('************************************')

             
with tab2:
    st.title("Upload CSV File")
    
    st.subheader('Instruction to note before uploading the files:')
    st.info("""
        1. No NaN values allowed.
        2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
        3. Check the spelling of the feature names.\n
        4. Feature values conventions:\n
            - Age: age of the patient [Years]
            - Sex: sex of the patient [0: Male, 1: Female]\n
            - ChestPainType: chest pain type [0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymtomatic, 3: Typical Angina]\n
            - RestingBP: resting Blood pressure [mm Hg]\n
            - Cholesterol: serum cholesterol [mm/dl]\n
            - FastingBS: fasting blood sugar [1: FastingBS > 120mg/dl, 0: otherwise]\n
            - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality, 2: Left Ventricular Hypertrophy]\n
            - MaxHR: maximum heart rate achieved [numeric value between 60 and 202]\n
            - ExerciseAngina: exercise-induced Angina [0: No, 1: Yes]\n
            - Oldpeak: oldpeak = ST [numeric value measured in depression]\n
            - ST_Slope: the slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]\n
            """)
    # Create file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        # Load all models
        model_lr = pickle.load(open('LogisticR.pkl', 'rb'))
        model_rf = pickle.load(open('RandomForest.pkl', 'rb'))
        model_dt = pickle.load(open('Dtree.pkl', 'rb'))
        model_svm = pickle.load(open('SVM.pkl', 'rb'))

        # Required features
        expected_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                             'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_features).issubset(input_data.columns):
            try:
                # Ensure column order is correct
                features_df = input_data[expected_features]

                # Perform predictions
                
                percentage = ((model_lr.predict(features_df) + model_rf.predict(features_df) + model_dt.predict(features_df) + model_svm.predict(features_df))/4)*100;
                input_data["Result (In %age Chance)"] = percentage 


                st.subheader("Bulk Predictions:")
                st.write(input_data)

                # Download link
                csv = input_data.to_csv(index=False)
                st.download_button(
                    label="Download Prediction CSV",
                    data=csv,
                    file_name="HeartPredictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("The uploaded CSV must have the correct 11 columns in the correct format.")
    else:
        st.info("Upload a CSV file to get predictions.")
         
with tab3:
    import plotly.express as px
    data = {'Decision Trees': 82.61, 'Logistic Regression':85.86, 'Random Forest':86.41, 'Support Vector Machine': 84.22}
    models=list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models,Accuracies)), columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies', x='Models')
    st.plotly_chart(fig)
