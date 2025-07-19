import streamlit as st

# Set page config
st.set_page_config(
    page_title="Multiple Linear Regression from Scratch",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title('Multiple Linear Regression from Scratch')
st.markdown("""
This app demonstrates how to implement **Multiple Linear Regression** from scratch using **Gradient Descent**.
Learn the mathematics behind the algorithm and see it in action!
""")

st.header("Understanding Multiple Linear Regression")

st.subheader("What is Multiple Linear Regression?")
st.markdown("""
Multiple Linear Regression is a statistical technique that models the relationship between:
- **One dependent variable (target)** - what we want to predict
- **Multiple independent variables (features)** - what we use to make predictions

The mathematical model is:
""")
st.latex(r"y = w_1x_1 + w_2x_2 + ... + w_nx_n + b")
st.markdown("Where:")
st.markdown("- $y$ = predicted value")
st.markdown("- $w_i$ = weights (coefficients)")
st.markdown("- $x_i$ = feature values")
st.markdown("- $b$ = bias term")

st.subheader("Gradient Descent Algorithm")
st.markdown("""
Gradient Descent is an optimization algorithm that finds the best weights by:
1. **Starting** with random weights
2. **Computing** the gradient (direction of steepest increase in error)
3. **Updating** weights in the opposite direction of the gradient
4. **Repeating** until convergence

The cost function (Mean Squared Error):
""")
st.latex(r"J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(h_w(x^{(i)}) - y^{(i)})^2")

st.subheader("Weight Update Rule")
st.latex(r"w_j := w_j - \alpha \frac{\partial}{\partial w_j}J(w,b)")
st.latex(r"b := b - \alpha \frac{\partial}{\partial b}J(w,b)")

st.markdown("Where $\\alpha$ is the learning rate.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by <a href='https://atherv.com'> Atherv Vidhate</a> | Multiple Linear Regression from Scratch</p>
</div>
""", unsafe_allow_html=True)