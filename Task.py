import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Iris Flower Classification App")
st.write("Yeh app Decision Tree Classifier ka use karke Iris dataset par model train aur evaluate karti hai.")

# Dataset load karna
iris = load_iris()
X = iris.data
Y = iris.target

# Sidebar par test size ke liye slider
test_size = st.sidebar.slider("Test Size (Percentage)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Model training
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Results display karna
st.subheader("Model Performance")
st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
st.dataframe(report)