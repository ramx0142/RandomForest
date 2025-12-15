import streamlit as st
import pickle
import pandas as pd

# 1. Load the trained model
# We use pickle because that is what you used in your notebook
try:
    with open('RandomForest.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'RandomForest.pkl' not found. Please download it from your notebook and place it in this folder.")
    st.stop()

# 2. App Title and Description
st.set_page_config(page_title="Car Purchase Predictor", page_icon="🚗")
st.title("🚗 Car Purchase Prediction")
st.write("Enter the customer's Age and Estimated Salary to predict if they will buy the car.")

# 3. User Inputs
col1, col2 = st.columns(2)

with col1:
    # Age input (Min 18 based on your data)
    age = st.number_input("Enter Age", min_value=18, max_value=100, value=25, step=1)

with col2:
    # Salary input
    salary = st.number_input("Enter Estimated Salary", min_value=0, value=50000, step=500)

# 4. Prediction Logic
if st.button("Predict Result"):
    # In your notebook (Page 2), you created a DataFrame with specific column names.
    # We must do the same here so the model recognizes the feature names.
    user_data = pd.DataFrame([[age, salary]], columns=['Age', 'EstimatedSalary'])
    
    try:
        prediction = model.predict(user_data)
        
        # Check result (1 = Buy, 0 = No Buy)
        st.markdown("---")
        if prediction[0] == 1:
            st.success("Prediction: The user **WILL BUY** the car. ✅")
        else:
            st.error("Prediction: The user **will NOT** buy the car. ❌")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.caption("Model: Random Forest Classifier")
