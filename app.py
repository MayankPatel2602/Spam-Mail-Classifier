import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open('model.pkl','rb'))
x_train = pickle.load(open('X_train.pkl','rb'))

st.header('Spam Mail Classifier')

message = st.text_input('Message in the Mail:')

input_mail=[message]

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_test_features = feature_extraction.fit_transform(x_train)
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)

if st.button('Classify the Mail'):
    if prediction==1:
        st.header('The Mail is a Spam mail')

    else:
        st.header('The Mail is a Genuine mail')
