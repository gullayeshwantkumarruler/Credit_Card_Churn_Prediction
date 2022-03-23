import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# To impute missing values
from sklearn.impute import SimpleImputer
import joblib
import pickle

best_model=joblib.load("pickle_tuned_adb3.pkl")
x_train=joblib.load("dataset.pkl")


def welcome():
    return "Welcome All"

@st.cache
def predict_churn(Customer_Age,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category):
    l=[Customer_Age,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category]
    cols=['Customer_Age','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category','Card_Category']
    data2=pd.DataFrame([l], columns=cols)
#     data2["Income_Category"].replace("abc", np.nan, inplace=True)
    data2 = pd.get_dummies( data2, columns=data2.select_dtypes(include=["object", "category"]).columns.tolist(), drop_first=True,)
    data2 = data2.reindex(columns = x_train.columns, fill_value=0)
    
    prediction=best_model.predict(data2)
    print(prediction)
    return int(round(prediction[0], 0))



def main():
    #st.title("Length of stay predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Credit Card User Churn Prediction</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    # Numerical
    Customer_Age= col1.number_input("Age of the customer", min_value=0, max_value=100)
    Months_on_book=col2.number_input("Months on Book", min_value=0, max_value=240)
#     Total_Relationship_Count=col1.selectbox("Number of Products used",(0,1,2,3,4,5,6))
#     Months_Inactive_12_mon=col2.selectbox("Number Months Inactive",(0,1,2,3,4,5,6,7,8,9,10,11,12))
#     Contacts_Count_12_mon=col1.selectbox("Number of interactions",(0,1,2,3,4,5,6,7,8,9,10,11,12))
    
    Total_Relationship_Count=col1.number_input("Relationship Count", min_value=0, max_value=6)
    Months_Inactive_12_mon=col2.number_input("Number of inactive months", min_value=0, max_value=12)
    Contacts_Count_12_mon=col1.number_input("Number of interactions", min_value=0, max_value=12)
    
    Credit_Limit=col2.number_input("Credit Limit", min_value=0, max_value=50000)
    Total_Revolving_Bal= col1.number_input("Revolving balance to be paid", min_value=0, max_value=50000)
    Avg_Open_To_Buy=col2.number_input("Money available for use", min_value=0, max_value=50000)
    Total_Amt_Chng_Q4_Q1=col1.number_input("Change in total amount", min_value=0, max_value=100)
    Total_Trans_Amt=col2.number_input("Total Transaction Amount", min_value=0, max_value=50000)
    Total_Trans_Ct=col1.number_input("Number of Transactions", min_value=0, max_value=200)
    Total_Ct_Chng_Q4_Q1=col2.number_input("change in total count", min_value=0, max_value=100)
    Avg_Utilization_Ratio=col1.number_input("utilization ratio", min_value=0, max_value=1)
    # categorical
    Gender= col2.selectbox("Gender",("F","M"))
    Dependent_count= col1.number_input("Number of Dependents", min_value=0, max_value=5)
    Education_Level= col2.selectbox("Education Level",("Graduate","High School","Uneducated","College","Post-Graduate","Doctorate"))
    Marital_Status=col1.selectbox("Martial Status",("Married","Single","Divorced"))
    Income_Category=col2.selectbox("Income Category",("Less than $40K","$40K - $60K","$80K - $120K","$60K - $80K","abc","$120K +"))
    Card_Category=col1.selectbox("Category of Card",("Blue","Silver","Gold","Platinum"))
    result=""
    if st.button("Predict"):        
        result=predict_churn(Customer_Age,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category)
        if result==1: 
            st.success('The Customer churned')
        else:
            st.success('The Customer did not churn')
            
    if st.button("About"):
        st.text("Credit Card Users Churn Prediction")
        # st.subheader("Data dictionary:")
        # st.text("Available Extra Rooms in Hospital: The number of rooms available during admission.")
        # st.text("Department: The department which will be treating the patient.")
        # st.text("Ward Facility Code: The code of the ward facility in which the patient will be admitted.")
    

if __name__=='__main__':
    main()
