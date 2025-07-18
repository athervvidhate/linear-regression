import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from multiple_lr import gradient_descent, predict, normalize_features

st.header("📁 Upload Your Own Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with your data"
)

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded dataset with {user_df.shape[0]} rows and {user_df.shape[1]} columns")
        
        # Display dataset info
        st.subheader("📋 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", user_df.shape)
            st.write("**Columns:**", list(user_df.columns))
        
        with col2:
            st.write("**Data Types:**")
            st.write(user_df.dtypes.value_counts())
        
        # Show sample data
        with st.expander("📊 View Dataset"):
            st.dataframe(user_df.head(10))
            st.write("**Dataset Statistics:**")
            st.dataframe(user_df.describe())
        
        # Column selection
        st.subheader("🎯 Select Features and Target")
        
        # Get numeric columns only
        numeric_columns = user_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("❌ Dataset must have at least 2 numeric columns (one for features, one for target)")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available numeric columns:**")
                st.write(numeric_columns)
                
                # Feature selection
                selected_features = st.multiselect(
                    "Select feature columns (independent variables):",
                    numeric_columns,
                    help="Choose the columns you want to use to predict the target"
                )
            
            with col2:
                # Target selection
                target_column = st.selectbox(
                    "Select target column (dependent variable):",
                    numeric_columns,
                    help="Choose the column you want to predict"
                )
            
            if selected_features and target_column and target_column not in selected_features:
                st.success("✅ Column selection complete!")
                
                # Model parameters
                st.subheader("⚙️ Model Parameters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    epochs = st.slider("Number of Epochs", 100, 5000, 1000, 100, key="user_epochs")
                with col2:
                    learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001, key="user_lr")
                with col3:
                    log_interval = st.slider("Log Interval", 10, 200, 100, 10, key="user_log")
                
                # Prepare data
                X_user = user_df[selected_features].to_numpy()
                y_user = user_df[target_column].to_numpy()
                
                # Remove rows with missing values
                mask = ~(np.isnan(X_user).any(axis=1) | np.isnan(y_user))
                X_user = X_user[mask]
                y_user = y_user[mask]
                
                if len(X_user) > 0:
                    # Feature scaling
                    X_user_scaled, mean_x_user, std_x_user = normalize_features(X_user)
                    y_user_scaled = (y_user - np.mean(y_user)) / (np.std(y_user) + 1e-9)
                    
                    # Initialize weights
                    w_user = np.random.randn(X_user_scaled.shape[1]) * 0.001
                    b_user = 0
                    
                    # Train model
                    if st.button("🚀 Train Model on Your Data"):
                        with st.spinner("Training in progress..."):
                            history_user, w_final_user, b_final_user = gradient_descent(
                                X_user_scaled, y_user_scaled, w_user, b_user,
                                epochs=epochs, alpha=learning_rate, log_interval=log_interval
                            )
                        
                        st.success("Training completed!")
                        
                        # Cost history
                        st.subheader("📈 Cost Function History")
                        cost_df_user = pd.DataFrame(list(history_user.items()), columns=['Epoch', 'Cost'])
                        
                        fig_cost_user = px.line(cost_df_user, x='Epoch', y='Cost',
                                                title="Cost Function Convergence")
                        fig_cost_user.update_layout(height=400)
                        st.plotly_chart(fig_cost_user, use_container_width=True)
                        
                        # Model performance
                        st.subheader("🎯 Model Performance")
                        y_pred_user_scaled = predict(X_user_scaled, w_final_user, b_final_user)
                        y_pred_user = y_pred_user_scaled * np.std(y_user) + np.mean(y_user)
                        
                        # Calculate metrics
                        mse_user = np.mean((y_pred_user - y_user) ** 2)
                        rmse_user = np.sqrt(mse_user)
                        mae_user = np.mean(np.abs(y_pred_user - y_user))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse_user:.2f}")
                        with col2:
                            st.metric("Root Mean Squared Error", f"{rmse_user:.2f}")
                        with col3:
                            st.metric("Mean Absolute Error", f"{mae_user:.2f}")
                        
                        # Final weights
                        st.subheader("🔧 Final Model Parameters")
                        weights_df_user = pd.DataFrame({
                            'Feature': selected_features,
                            'Weight': w_final_user,
                            'Scaled Weight': w_final_user * std_x_user
                        })
                        st.dataframe(weights_df_user)
                        st.write(f"**Bias (b):** {b_final_user:.6f}")
                        
                        # Predictions vs Actual plot
                        st.subheader("📊 Predictions vs Actual Values")
                        
                        # Sample data for visualization
                        sample_size_user = min(100, len(y_user))
                        sample_indices_user = np.random.choice(len(y_user), sample_size_user, replace=False)
                        
                        fig_pred_user = go.Figure()
                        fig_pred_user.add_trace(go.Scatter(
                            x=y_user[sample_indices_user],
                            y=y_pred_user[sample_indices_user],
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='green', size=8)
                        ))
                        
                        # Perfect prediction line
                        min_val_user = min(y_user[sample_indices_user].min(), y_pred_user[sample_indices_user].min())
                        max_val_user = max(y_user[sample_indices_user].max(), y_pred_user[sample_indices_user].max())
                        fig_pred_user.add_trace(go.Scatter(
                            x=[min_val_user, max_val_user],
                            y=[min_val_user, max_val_user],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_pred_user.update_layout(
                            title="Predictions vs Actual Values",
                            xaxis_title="Actual Values",
                            yaxis_title="Predicted Values",
                            height=500
                        )
                        st.plotly_chart(fig_pred_user, use_container_width=True)             
                else:
                    st.error("❌ No valid data remaining after removing missing values")
                    
            elif target_column in selected_features:
                st.error("❌ Target column cannot be included in features")
            else:
                st.warning("⚠️ Please select at least one feature column and a target column")
                
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")