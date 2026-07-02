import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("randomforest.pkl", "rb"))

st.title("Raisin Class Prediction 🍇 (RandomForest)")

area = st.number_input("Area")
major = st.number_input("MajorAxisLength")
minor = st.number_input("MinorAxisLength")
ecc = st.number_input("Eccentricity")
convex = st.number_input("ConvexArea")
extent = st.number_input("Extent")
perimeter = st.number_input("Perimeter")

if st.button("Predict"):
    data = np.array([[area, major, minor, ecc, convex, extent, perimeter]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("Kecimen 🍇")
    else:
        st.success("Besni 🍇")