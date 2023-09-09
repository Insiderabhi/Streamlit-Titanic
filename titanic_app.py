
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the Titanic dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("C:\\Users\\ABHISHEKXD\\Downloads\\Ds\\vscodes\\stream_Titanic\\trainn.csv")
    return data

data = load_data()

# Data Cleaning: Exclude non-relevant columns ('Name', 'Ticket', 'Cabin')
data_cleaned = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values by imputing with the mean for numerical columns
data_cleaned['Age'].fillna(data_cleaned['Age'].mean(), inplace=True)
data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].mode()[0], inplace=True)

# Feature Engineering: One-Hot Encoding for categorical columns ('Sex' and 'Embarked')
categorical_columns = ['Sex', 'Embarked']  # Specify the columns you want to encode
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Split data into train and test sets
X = data_encoded.drop(['Survived'], axis=1)
y = data_encoded['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display model accuracy
st.sidebar.subheader('Model Performance')
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write('Accuracy:', accuracy)

# Create a form for user input
st.sidebar.subheader('Predict Survival')

# Input fields for user to enter feature values
st.sidebar.write('Enter the following information:')
passenger_id = st.sidebar.text_input('PassengerId', '1')  # Example default value
pclass = st.sidebar.slider('Pclass (1st, 2nd, 3rd)', 1, 3, 2)
age = st.sidebar.slider('Age', 0, 100, 30)
sib_sp = st.sidebar.slider('SibSp (Number of Siblings/Spouses Aboard)', 0, 8, 0)
parch = st.sidebar.slider('Parch (Number of Parents/Children Aboard)', 0, 6, 0)
fare = st.sidebar.slider('Fare', 0, 512, 50)
sex_male = st.sidebar.checkbox('Is the passenger male?')
embarked_Q = st.sidebar.checkbox('Embarked from Queenstown (Q)?')
embarked_S = st.sidebar.checkbox('Embarked from Southampton (S)?')

# Prepare the input data with one-hot encoding for 'Sex' and 'Embarked'
input_data = pd.DataFrame({
    'PassengerId': [passenger_id],
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sib_sp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex_male else 0],
    'Embarked_Q': [1 if embarked_Q else 0],
    'Embarked_S': [1 if embarked_S else 0],
})

# Make a prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)[0]
    prediction_text = 'Survived' if prediction == 1 else 'Not Survived'
    st.sidebar.write(f'Prediction: {prediction_text}')
