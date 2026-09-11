import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    st.success("Predicted Flower: " + flowers[pred[0]])

    # Visualization
    iris = load_iris()

    fig, ax = plt.subplots()

    ax.scatter(
        iris.data[:, 0],
        iris.data[:, 2],
        label="Iris Dataset"
    )

    ax.scatter(
        sl,
        pl,
        s=150,
        marker="*",
        label="Prediction"
    )

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")
    ax.set_title("Iris Dataset Visualization")
    ax.legend()

    st.pyplot(fig)