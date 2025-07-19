import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

st.set_page_config(
    page_title="Upload Your Own Dataset",
    page_icon="📁",
    layout="wide"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from multiple_lr import gradient_descent, predict, normalize_features, train_test_split

st.header("Upload Your Own Dataset")

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
                    # Train/test split
                    X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X_user, y_user, test_size=0.2)
                    
                    # Feature scaling (fit only on train, apply to both)
                    X_train_scaled_user, mean_x_user, std_x_user = normalize_features(X_train_user)
                    X_test_scaled_user = (X_test_user - mean_x_user) / (std_x_user + 1e-9)
                    
                    y_train_scaled_user = (y_train_user - np.mean(y_train_user)) / (np.std(y_train_user) + 1e-9)
                    mean_y_train_user = np.mean(y_train_user)
                    std_y_train_user = np.std(y_train_user)
                    
                    # Initialize weights
                    w_user = np.random.randn(X_train_scaled_user.shape[1]) * 0.001
                    b_user = 0
                    
                    # Train model
                    if st.button("🚀 Train Model on Your Data"):
                        with st.spinner("Training in progress..."):
                            history_user, w_final_user, b_final_user = gradient_descent(
                                X_train_scaled_user, y_train_scaled_user, w_user, b_user,
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
                        # Predict on train and test
                        y_train_pred_scaled_user = predict(X_train_scaled_user, w_final_user, b_final_user)
                        y_train_pred_user = y_train_pred_scaled_user * std_y_train_user + mean_y_train_user
                        y_test_pred_scaled_user = predict(X_test_scaled_user, w_final_user, b_final_user)
                        y_test_pred_user = y_test_pred_scaled_user * std_y_train_user + mean_y_train_user
                        
                        # Calculate metrics
                        mse_train_user = np.mean((y_train_pred_user - y_train_user) ** 2)
                        rmse_train_user = np.sqrt(mse_train_user)
                        
                        mse_test_user = np.mean((y_test_pred_user - y_test_user) ** 2)
                        rmse_test_user = np.sqrt(mse_test_user)
                        
                        st.write("#### Training Set Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Train MSE", f"{mse_train_user:.2f}")
                        with col2:
                            st.metric("Train RMSE", f"{rmse_train_user:.2f}")
                        
                        st.write("#### Test Set Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Test MSE", f"{mse_test_user:.2f}")
                        with col2:
                            st.metric("Test RMSE", f"{rmse_test_user:.2f}")
                        
                        # Calculate improvement over baseline
                        baseline_rmse_user = np.std(y_test_user)
                        improvement_percentage_user = ((baseline_rmse_user - rmse_test_user) / baseline_rmse_user) * 100
                        
                        if improvement_percentage_user > 5:
                            st.success(f"🎉 **Great job!** We improved our model by **{improvement_percentage_user:.1f}%** over the baseline (predicting the mean).")
                        elif improvement_percentage_user > 0:
                            st.info(f"✅ **Good!** We improved our model by **{improvement_percentage_user:.1f}%** over the baseline.")
                        else:
                            st.warning(f"⚠️ **Model needs improvement.** Our model is **{abs(improvement_percentage_user):.1f}%** worse than just predicting the mean.")
                        
                        # Final weights
                        st.subheader("🔧 Final Model Parameters")
                        weights_df_user = pd.DataFrame({
                            'Feature': selected_features,
                            'Weight': w_final_user,
                            'Scaled Weight': w_final_user * std_x_user
                        })
                        st.dataframe(weights_df_user)
                        st.write(f"**Bias (b):** {b_final_user:.6f}")
                        
                        # Predictions vs Actual plot (test set)
                        st.subheader("📊 Predictions vs Actual Values (Test Set)")
                        
                        # Sample data for visualization
                        sample_size_user = min(100, len(y_test_user))
                        sample_indices_user = np.random.choice(len(y_test_user), sample_size_user, replace=False)
                        
                        fig_pred_user = go.Figure()
                        fig_pred_user.add_trace(go.Scatter(
                            x=y_test_user[sample_indices_user],
                            y=y_test_pred_user[sample_indices_user],
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='green', size=8)
                        ))
                        
                        # Perfect prediction line
                        min_val_user = min(y_test_user[sample_indices_user].min(), y_test_pred_user[sample_indices_user].min())
                        max_val_user = max(y_test_user[sample_indices_user].max(), y_test_pred_user[sample_indices_user].max())
                        fig_pred_user.add_trace(go.Scatter(
                            x=[min_val_user, max_val_user],
                            y=[min_val_user, max_val_user],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_pred_user.update_layout(
                            title="Predictions vs Actual Values (Test Set)",
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by <a href='https://atherv.com'> Atherv Vidhate</a> | Multiple Linear Regression from Scratch</p>
</div>
""", unsafe_allow_html=True)