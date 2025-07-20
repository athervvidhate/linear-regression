import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from multiple_lr import gradient_descent, predict, normalize_features, train_test_split

st.header("California Housing Dataset Demo")

# Load and display dataset
@st.cache_data
def read_housing_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'california_housing_train.csv')
    return pd.read_csv(csv_path)

df = read_housing_dataset()
prediction_features = df.columns[:-1]

st.subheader("Dataset Overview")
st.write(f"**Dataset Shape:** {df.shape}")
st.write(f"**Features:** {list(df.columns)}")

# Show sample data
with st.expander("View Dataset"):
    st.write("**First 50 rows of the dataset:**")
    st.dataframe(df.head(50))
    st.write("**Dataset Statistics:**")
    st.dataframe(df.describe())

# Parameter controls
st.subheader("Model Parameters")

selected_features = st.multiselect(
                    "Select model features to predict housing price:",
                    prediction_features,
                    help="Choose the columns you want to use to predict the price",
                    default=prediction_features
                )

col1, col2, col3 = st.columns(3)

with col1:
    epochs = st.slider("Number of Epochs", 100, 5000, 1000, 100)
with col2:
    learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001)
with col3:
    log_interval = st.slider("Log Interval", 10, 200, 100, 10)

# Prepare data
X = df[selected_features].to_numpy()
y = df['median_house_value'].to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Add explanation of train/test split
with st.expander("What is train/test split and how does our implementation work?"):
    st.write("""
    **What is Train/Test Split?**
    
    Train/test split is a fundamental concept in machine learning that helps us evaluate how well our model will perform on new, unseen data. Here's why it's important:
    
    - **Training Set (80% of data)**: Used to train the model and learn the relationships between features and target
    - **Test Set (20% of data)**: Used to evaluate the model's performance on data it has never seen before
    
    **Why Do We Need It?**
    
    Without a test set, we would only know how well the model performs on the data it was trained on. This can lead to:
    - **Overfitting**: When a model performs very well on training data but poorly on new data
    - **Overly optimistic performance estimates**: The model might memorize the training data instead of learning generalizable patterns
    
    **Custom Implementation:**
    
    Instead of using scikit-learn's `train_test_split`, I implemented my own version:
    
    ```python
    def shuffle_dataset(X, y):
        length = np.arange(X.shape[0])
        np.random.shuffle(length)
        return X[length], y[length]
    
    def train_test_split(X, y, test_size=0.5, shuffle=True):
        if shuffle:
            X, y = shuffle_dataset(X, y)
        if test_size < 1:
            train_ratio = len(y) - int(len(y) * test_size)
            X_train, X_test = X[:train_ratio], X[train_ratio:]
            y_train, y_test = y[:train_ratio], y[train_ratio:]
            return X_train, X_test, y_train, y_test
    ```
    
    **How It Works:**
    1. **Shuffle the data** to ensure random sampling
    2. **Split based on test_size**: If test_size=0.2, we use 80% for training and 20% for testing
    3. **Return separate arrays** for training and testing features and targets
    """)

# Feature scaling (fit only on train, apply to both)
X_train_scaled, mean_x, std_x = normalize_features(X_train)
X_test_scaled = (X_test - mean_x) / (std_x + 1e-9)

y_train_scaled = (y_train - np.mean(y_train)) / (np.std(y_train) + 1e-9)
mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)

# Initialize random weights
w = np.random.randn(X_train_scaled.shape[1]) * 0.001
b = 0

if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history, w_final, b_final = gradient_descent(
            X_train_scaled, y_train_scaled, w, b, 
            epochs=epochs, alpha=learning_rate, log_interval=log_interval
        )
    
    st.success("Training completed!")
    
    # Cost history plot
    st.subheader("Cost Function History")
    cost_df = pd.DataFrame(list(history.items()), columns=['Epoch', 'Cost'])
    
    fig_cost = px.line(cost_df, x='Epoch', y='Cost', 
                        title="Cost Function Convergence",
                        labels={'Cost': 'Mean Squared Error'})
    fig_cost.update_layout(height=400)
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Model performance
    st.subheader("Model Performance")
    # Predict on train and test
    y_train_pred_scaled = predict(X_train_scaled, w_final, b_final)
    y_train_pred = y_train_pred_scaled * std_y_train + mean_y_train
    y_test_pred_scaled = predict(X_test_scaled, w_final, b_final)
    y_test_pred = y_test_pred_scaled * std_y_train + mean_y_train
    
    # Calculate metrics
    mse_train = np.mean((y_train_pred - y_train) ** 2)
    rmse_train = np.sqrt(mse_train)
    
    mse_test = np.mean((y_test_pred - y_test) ** 2)
    rmse_test = np.sqrt(mse_test)
    
    st.write("#### Training Set Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train MSE", f"{mse_train:.2f}")
    with col2:
        st.metric("Train RMSE", f"{rmse_train:.2f}")
    
    st.write("#### Test Set Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test MSE", f"{mse_test:.2f}")
    with col2:
        st.metric("Test RMSE", f"{rmse_test:.2f}")

    # Calculate improvement over baseline (predicting mean)
    baseline_rmse = np.std(y_test)  # RMSE if we always predict the mean
    improvement_percentage = ((baseline_rmse - rmse_test) / baseline_rmse) * 100
    
    if improvement_percentage > 5:  # Significant improvement threshold
        st.success(f"🎉 **Great job!** We improved our model by **{improvement_percentage:.1f}%** over the baseline (predicting the mean).")
    elif improvement_percentage > 0:
        st.info(f"✅ **Good!** We improved our model by **{improvement_percentage:.1f}%** over the baseline.")
    else:
        st.warning(f"⚠️ **Model needs improvement.** Our model is **{abs(improvement_percentage):.1f}%** worse than just predicting the mean.")

    with st.expander("Doesn't the test error still seem kind of...huge?"):
        st.write(f"""
                > *I thought we were trying to *minimize* error.*
                    
                At first glance, the values for these errors seem to be very, very large. 
                However, we have to remember the *squared* part of Mean Squared Error, so the units are squared too.
                
                Since we are predicting the median home price, the units for MSE is dollars$$^{2}$$.
                That's why we use the Root Mean Squared Error to give a more interpretable metric.
                
                Looking at the RMSE, we see \${rmse_test:,.2f}. That means that on average, our model is off on its predictions by about \${int(round(rmse_test, -3)):,}.

                > *But that still seems like a lot! How can this be a good model?*
                
                I hear you. But think about this. If you open up the dropdown above, you can look at the standard deviation of the median_house_value column,
                which again, is what we are trying to predict.

                The standard deviation is about 115,984. This model's RMSE is **significantly** less than the standard deviation of the target variable. This means:

                1. The model is learning something useful!

                2. If we always predicted the mean, the RMSE would be about 115,984.

                3. Since model's RMSE is much lower, it's capturing meaningful relationships in the data.

                If we wanted to further reduce the RMSE, we could try more features, regularization, or implement other more complex models.
                
                """)

    # Final weights
    st.subheader("Final Model Parameters (Feature Importance)")
    feature_names = df.drop(['median_house_value'], axis=1).columns
    
    weights_df = pd.DataFrame({
        'Feature': selected_features,
        'Weight': w_final,
        'Scaled Weight': w_final * std_x
    })
    st.dataframe(weights_df)
    st.write(f"**Bias (b):** {b_final}")
    
    # Predictions vs Actual plot (test set)
    st.subheader("Predictions vs Actual Values (Test Set)")
    
    # Sample data for visualization (first 100 points)
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test[sample_indices], 
        y=y_test_pred[sample_indices],
        mode='markers',
        name='Predictions',
        marker=dict(color='green', size=8, opacity=0.5)
    ))
    
    # Perfect prediction line
    min_val = min(y_test[sample_indices].min(), y_test_pred[sample_indices].min())
    max_val = max(y_test[sample_indices].max(), y_test_pred[sample_indices].max())
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig_pred.update_layout(
        title="Predictions vs Actual Values (Test Set)",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True) 

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by <a href='https://atherv.com'> Atherv Vidhate</a> | Multiple Linear Regression from Scratch</p>
</div>
""", unsafe_allow_html=True)