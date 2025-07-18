import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from multiple_lr import gradient_descent, predict, normalize_features

st.header("California Housing Dataset Demo")

# Load and display dataset
@st.cache_data
def read_housing_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'california_housing_train.csv')
    return pd.read_csv(csv_path)

df = read_housing_dataset()

st.subheader("Dataset Overview")
st.write(f"**Dataset Shape:** {df.shape}")
st.write(f"**Features:** {list(df.columns)}")

# Show sample data
with st.expander("View Dataset"):
    st.dataframe(df.head(10))
    st.write("**Dataset Statistics:**")
    st.dataframe(df.describe())

# Parameter controls
st.subheader("Model Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    epochs = st.slider("Number of Epochs", 100, 5000, 1000, 100)
with col2:
    learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001)
with col3:
    log_interval = st.slider("Log Interval", 10, 200, 100, 10)

# Prepare data
X = df.drop(['median_house_value'], axis=1).to_numpy()
y = df['median_house_value'].to_numpy()

# Feature scaling
X_scaled, mean_x, std_x = normalize_features(X)
y_scaled = (y - np.mean(y)) / (np.std(y) + 1e-9)

# Initialize weights
w = np.random.randn(X_scaled.shape[1]) * 0.001
b = 0

# Run gradient descent
if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history, w_final, b_final = gradient_descent(
            X_scaled, y_scaled, w, b, 
            epochs=epochs, alpha=learning_rate, log_interval=log_interval
        )
    
    # Display results
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
    y_pred_scaled = predict(X_scaled, w_final, b_final)
    y_pred = y_pred_scaled * np.std(y) + np.mean(y)
    
    # Calculate metrics
    mse = np.mean((y_pred - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{rmse:.2f}")
    with col3:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    

    #TODO: After we allow for feature selection, only show this if it is pretty large
    with st.expander("Doesn't this kind of...huge?"):
        st.write(f"""
                > I thought we were trying to *minimize* error.
                 
                At first glance, the values for these errors seem to be very, very large. 
                However, we have to remember the *squared* part of Mean Squared Error, so the units are squared too.
                
                Since we are predicting the median home price, the units for MSE is dollars$$^{2}$$.
                That's why we use the Root Mean Squared Error to give a more interpretable metric.
                
                Looking at the RMSE, we see \${rmse:.2f}. That means that on average, our model is off on its predictions by about \${int(round(rmse,-3))}.

                > *But that still seems like a lot! How can this be a good model?*
                
                I hear you. But think about this. If you open up the dropdown above, you can look at the standard deviation of the median_house_value column,
                which again, is what we are trying to predict.

                The standard deviation is about 115,984. This model's RMSE is **significantly** less than the standard deviation of the target variable. This means:

                1. The model is learning something useful!

                2. If we always predicted the mean, the RMSE would be about 115,984.

                3. Since model's RMSE is much lower, it's capturing meaningful relationships in the data.

                If we wanted to further reduce the RMSE, we could try more features, regularization, or implement other, more complex models.
                
                """)

    # Feature importance plot
    st.subheader("Feature Importance")
    feature_names = df.drop(['median_house_value'], axis=1).columns
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': w_final,
        'Scaled Weight': w_final * std_x
    })
    fig_importance = px.bar(
        weights_df, 
        x='Feature', 
        y='Scaled Weight',
        title="Feature Importance (Scaled Weights)",
        color='Scaled Weight',
        color_continuous_scale='RdBu'
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)

    # Final weights
    st.subheader("Final Model Parameters")
    feature_names = df.drop(['median_house_value'], axis=1).columns
    
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': w_final,
        'Scaled Weight': w_final * std_x
    })
    st.dataframe(weights_df)
    st.write(f"**Bias (b):** {b_final}")
    
    # Predictions vs Actual plot
    st.subheader("Predictions vs Actual Values")
    
    # Sample data for visualization (first 100 points)
    sample_size = min(100, len(y))
    sample_indices = np.random.choice(len(y), sample_size, replace=False)
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y[sample_indices], 
        y=y_pred[sample_indices],
        mode='markers',
        name='Predictions',
        marker=dict(color='green', size=8, opacity=0.5)
    ))
    
    # Perfect prediction line
    min_val = min(y[sample_indices].min(), y_pred[sample_indices].min())
    max_val = max(y[sample_indices].max(), y_pred[sample_indices].max())
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig_pred.update_layout(
        title="Predictions vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)
