# 📊 Multiple Linear Regression Tutorial

A comprehensive Streamlit application that teaches Multiple Linear Regression from scratch using Gradient Descent. This interactive tutorial includes educational content, hands-on demonstrations, and the ability to upload your own datasets.

## 🚀 Features

### 📚 Educational Content
- **Theory & Explanation**: Learn the mathematics behind Multiple Linear Regression
- **Gradient Descent Algorithm**: Understand how optimization works
- **Mathematical Formulas**: Interactive LaTeX equations explaining the concepts

### 🏠 California Housing Demo
- **Interactive Parameters**: Adjust epochs, learning rate, and log interval
- **Real-time Training**: Watch the model train with live cost function visualization
- **Performance Metrics**: MSE, RMSE, and MAE calculations
- **Model Visualization**: Predictions vs actual values plots
- **Feature Importance**: Bar charts showing feature weights

### 📁 Upload Your Data
- **CSV Upload**: Upload your own datasets
- **Column Selection**: Choose which features to use and what to predict
- **Interactive Training**: Train models on your data with custom parameters
- **Comprehensive Visualizations**: Multiple charts and metrics for your results

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd linear-regression
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
linear-regression/
├── streamlit_app.py          # Main Streamlit application
├── multiple_lr.py           # Multiple linear regression implementation
├── california_housing_train.csv  # Sample dataset
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## 🎯 How to Use

### 1. Theory & Explanation
- Navigate to the "📚 Theory & Explanation" section
- Learn about Multiple Linear Regression concepts
- Understand the mathematical foundations
- Review the Gradient Descent algorithm

### 2. California Housing Demo
- Go to "🏠 California Housing Demo"
- Adjust model parameters using the sliders
- Click "🚀 Train Model" to start training
- Observe the cost function convergence
- Analyze model performance and visualizations

### 3. Upload Your Data
- Navigate to "📁 Upload Your Data"
- Upload a CSV file with your dataset
- Select feature columns (independent variables)
- Choose a target column (dependent variable)
- Train the model and analyze results

## 📊 Sample Dataset

The app includes the California Housing dataset as a demonstration. This dataset contains:
- **Features**: Various housing characteristics (median income, housing age, etc.)
- **Target**: Median house value
- **Size**: Multiple features for comprehensive demonstration

## 🔧 Technical Details

### Multiple Linear Regression Model
The mathematical model implemented is:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Cost Function
Mean Squared Error (MSE):
```
J(w,b) = (1/2m) * Σ(h_w(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

### Gradient Descent
Weight update rules:
```
w_j := w_j - α * (∂/∂w_j)J(w,b)
b := b - α * (∂/∂b)J(w,b)
```

## 🎨 Visualizations

The app provides several interactive visualizations:
- **Cost Function History**: Shows convergence over epochs
- **Predictions vs Actual**: Scatter plot with perfect prediction line
- **Feature Importance**: Bar chart of scaled weights
- **Model Performance Metrics**: MSE, RMSE, MAE

## 🚀 Customization

### Adding New Datasets
1. Place your CSV file in the project directory
2. Update the file path in the code if needed
3. Ensure your data has numeric columns for features and target

### Modifying Parameters
- **Epochs**: Number of training iterations (100-5000)
- **Learning Rate**: Step size for gradient descent (0.001-0.1)
- **Log Interval**: How often to record cost (10-200)

## 📈 Performance Tips

- **Feature Scaling**: The app automatically normalizes features for better convergence
- **Learning Rate**: Start with 0.01 and adjust based on convergence
- **Epochs**: More epochs generally improve performance but increase training time
- **Data Quality**: Ensure your dataset has sufficient numeric features
