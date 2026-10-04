import streamlit as st
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="📢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Machine Learning Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("complaint_classifier.joblib")


model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">📢 Complaint Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Complaint Classification System</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Complaint Input
# --------------------------------------------------

st.subheader("📝 Enter Your Complaint")

text = st.text_area(
    "Complaint",
    placeholder="Example: I was charged twice for the same transaction...",
    height=150,
    label_visibility="collapsed"
)

# --------------------------------------------------
# Classification
# --------------------------------------------------

if st.button(
    "🔍 Classify Complaint",
    use_container_width=True
):

    # Check for blank input
    if not text.strip():

        st.warning(
            "⚠️ Please enter a complaint before clicking the classify button."
        )

    else:

        # Make prediction
        prediction = model.predict([text])[0]

        # Display result
        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction</h3>
                <h2>📌 {prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Information Section
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About this system"):

    st.write(
        """
        This application uses a machine learning model to automatically
        classify customer complaints into predefined categories.

        Enter a complaint in the text box and click
        **Classify Complaint** to receive the predicted category.
        """
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Complaint Classification System | Powered by Machine Learning"
)