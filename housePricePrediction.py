import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.write("""
# Simple House price Prediction App

This app **Predicts the price of House** according to features!
         
It is using **Random Forest Regression** algorithm.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    bedrooms = st.sidebar.slider('Bedrooms',1,15,2,1,key=0)

    bathrooms = st.sidebar.slider('Bathrooms',1,12,2,1,key=1)

    sqft_living = st.sidebar.slider('Area of Living (Sqft)',300,10000,560,50,key=2)

    sqft_land = st.sidebar.slider('Area of Land (Sqft)',400,20000,700,100,key=3)

    floors = st.sidebar.slider('Floors',1,6,1,1,key=4)

    st.sidebar.write("<p style='margin-bottom: 2px;'>Waterfront</p>", unsafe_allow_html=True)
    waterfront_input = st.sidebar.radio("Select Yes or No", ["Yes", "No"], index=1)
    waterfront = 1 if waterfront_input == 'Yes' else 0

    st.sidebar.write("<p style='margin-bottom: 2px;'>View</p>", unsafe_allow_html=True)
    view_input = st.sidebar.radio("Select Yes or No", ["Yes", "No"], index=1, key='Yes')
    view = 1 if view_input == 'Yes' else 0

    st.sidebar.write("<p style='margin-bottom: 2px;'>Condition</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>1: Poor</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>2: Fair</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>3: Average</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>4: Good</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>5: Excellent</p>", unsafe_allow_html=True)
    condition = st.sidebar.slider('Select from above',1,5,3,1,key=5)

    st.sidebar.write("<p style='margin-bottom: 2px;'>Grades</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>(1-3): Poor Construction</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>(4-7): Standard Construction</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>(8-9): Good to Very Good Quality</p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='margin-bottom: 2px;'>(10-12): High Quality with Luxury Features</p>", unsafe_allow_html=True)
    grade = st.sidebar.slider('Select from above',1,12,6,1,key=6)

    sqft_above = st.sidebar.slider('Sqft Above', 300, 10000, 560, 50,key=7)

    sqft_basement = st.sidebar.slider('Sqft Basement', 0, 5000, 0, 50,key=8)

    yr_built = st.sidebar.slider('Year Built', 1900, 2022, 1970, 1,key=9)

    yr_renovated = st.sidebar.slider('Year Renovated', 1900, 2022, 1970, 1, key=10)

    sqft_living15 = st.sidebar.slider("Average interior square footage of the 15 nearest neighbors' living spaces", 300, 10000, 560, 50, key=11)
    
    sqft_lot15 = st.sidebar.slider('The average square footage of the land lots of the 15 nearest neighbors', 400, 20000, 700, 100, key=12)

    zipcode_options = ['98002', '98003', '98004', '98005', '98006', '98007', '98008', '98010', '98011', '98014',
                       '98019', '98022', '98023', '98024', '98027', '98028', '98029', '98030', '98031', '98032',
                       '98033', '98034', '98038', '98039', '98040', '98042', '98045', '98052', '98053', '98055',
                       '98056', '98058', '98059', '98065', '98070', '98072', '98074', '98075', '98077', '98092',
                       '98102', '98103', '98105', '98106', '98107', '98108', '98109', '98112', '98115', '98116',
                       '98117', '98118', '98119', '98122', '98125', '98126', '98133', '98136', '98144', '98146',
                       '98148', '98155', '98166', '98168', '98177', '98178', '98188', '98198', '98199']

    selected_zipcode = st.sidebar.selectbox('Zipcode', zipcode_options)

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_land,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15,
        f'zipcode_{selected_zipcode}': True,  # Set the selected zipcode to True
    }
    # Set all zipcodes to False
    for zipcode_option in zipcode_options:
        data[f'zipcode_{zipcode_option}'] = False

    # Set the selected zipcode to True
    data[f'zipcode_{selected_zipcode}'] = True
    
    features = pd.DataFrame(data, index=[0])
    return features

user_features = user_input_features()

st.subheader('User Input Parameters')
st.write(user_features)

#loading dataset

@st.cache_data
def load_dataset():
    return pd.read_csv('house_data.csv')

df = load_dataset()
y=df['price']
X=df.drop("price",axis=1)

#splitting dataset
@st.cache_data
def split_dataset():
    return train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_test, y_train, y_test = split_dataset()


#model Training
@st.cache_data
def train_model():
    rfregressor = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=4,
                                        max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=10)
    rfregressor.fit(X_train, y_train)
    return rfregressor

model = train_model()

#Prediction

@st.cache_data
def make_prediction(features):
    # Load the model inside the function to avoid UnhashableParamError
    model = train_model()

    # Extract feature names from the training data
    feature_names = X.columns.tolist()

    # Ensure the input features have the same order as the training data
    features = features[feature_names]

    return model.predict(features)

y_pred = make_prediction(X_test)

prediction = make_prediction(user_features)

st.write('## Prediction')
st.write(f'Predicted House Price: **${int(prediction[0]):,}**')

st.write("## Data and Result Analysis")
st.subheader('Dataframe Sample')
st.write(df.head(10))

#Model Analysis

def analyze_model(X_test, y_test, model):
    model_params = (model.get_params(),)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

mse, rmse, mae, r2 = analyze_model(X_test, y_test, model)

# Display the mathematical result analysis
st.subheader('Mathematical Result Analysis')
st.write(f'R-squared (R2): **{r2:.2f}**')
st.write(f'Mean Squared Error (MSE): **{mse:.2f}**')
st.write(f'Root Mean Squared Error (RMSE): **{rmse:.2f}**')
st.write(f'Mean Absolute Error (MAE): **{mae:.2f}**')


# Add space to push the footer to the bottom
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/css/all.min.css">', unsafe_allow_html=True)

footer_html = """
---
### Contact Me
For inquiries or support, please contact me at [maradiyavignesh2004@gmail.com](mailto:maradiyavignesh2004@gmail.com).

[<i class="fab fa-github" style="color: white; font-size: 25px;"></i>](https://github.com/vigneshmaradiya)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[<i class="fab fa-linkedin" style="color: white; font-size: 25px;"></i>](https://www.linkedin.com/in/vignesh-maradiya/)

&copy; 2024 Vignesh Maradiya. All rights reserved. Developed by Vignesh Maradiya.
"""

st.markdown(footer_html, unsafe_allow_html=True)