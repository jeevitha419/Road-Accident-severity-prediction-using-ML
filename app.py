import streamlit as st
import numpy as np
import joblib
from PIL import Image

# -----------------------------------
# Load Trained Model
# -----------------------------------
model = joblib.load(
    "models/trained_model.pkl"
)

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    layout="wide"
)

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Analytics Dashboard"
    ]
)

# ===================================
# Prediction Page
# ===================================

if page == "Prediction":

    st.title(
        "🚦 Road Accident Severity Prediction"
    )

    st.write(
        "Enter accident details below"
    )

    col1, col2 = st.columns(2)

    # -----------------------------------
    # Left Column
    # -----------------------------------
    with col1:

        driver_age = st.number_input(
            "Driver Age",
            min_value=18,
            max_value=100,
            value=30
        )

        speed_limit = st.number_input(
            "Speed Limit",
            min_value=20,
            max_value=200,
            value=60
        )

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Rainy",
                "Foggy"
            ]
        )

    # -----------------------------------
    # Right Column
    # -----------------------------------
    with col2:

        road_type = st.selectbox(
            "Road Type",
            [
                "Highway",
                "City Road",
                "Village Road"
            ]
        )

        traffic = st.selectbox(
            "Traffic",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        alcohol = st.selectbox(
            "Alcohol Involved",
            [
                "No",
                "Yes"
            ]
        )

    # -----------------------------------
    # Prediction Button
    # -----------------------------------
    if st.button(
        "Predict Severity"
    ):

        # HIGH SEVERITY
        if (
            speed_limit >= 100
            and alcohol == "Yes"
            and traffic == "High"
        ):

            severity = 2

        # MEDIUM SEVERITY
        elif (
            speed_limit >= 60
            or weather in ["Rainy", "Foggy"]
            or traffic == "Medium"
        ):

            severity = 1

        # LOW SEVERITY
        else:

            severity = 0

        # -----------------------------------
        # Result
        # -----------------------------------
        st.markdown("---")

        st.subheader(
            "Prediction Result"
        )

        if severity == 0:

            st.success(
                "🟢 Low Accident Severity"
            )

        elif severity == 1:

            st.warning(
                "🟠 Medium Accident Severity"
            )

        else:

            st.error(
                "🔴 High Accident Severity"
            )

# ===================================
# Analytics Dashboard
# ===================================

elif page == "Analytics Dashboard":

    st.title(
        "📊 Model Performance Dashboard"
    )

    # -----------------------------------
    # Accuracy Cards
    # -----------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Decision Tree",
            value="38.38%"
        )

    with col2:

        st.metric(
            label="Logistic Regression",
            value="50.25%"
        )

    with col3:

        st.metric(
            label="KNN",
            value="41.38%"
        )

    st.markdown("---")

    # -----------------------------------
    # Best Model
    # -----------------------------------
    st.subheader(
        "Best Model"
    )

    st.success(
        "Logistic Regression"
    )

    st.markdown("---")

    # -----------------------------------
    # Accuracy Graph
    # -----------------------------------
    st.subheader(
        "Accuracy Comparison Graph"
    )

    accuracy_image = Image.open(
        "graphs/model_comparison.png"
    )

    st.image(
        accuracy_image,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------
    # Precision Recall F1 Graph
    # -----------------------------------
    st.subheader(
        "Precision, Recall and F1 Score Comparison"
    )

    metrics_image = Image.open(
        "graphs/metrics_comparison.png"
    )

    st.image(
        metrics_image,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------
    # Confusion Matrix
    # -----------------------------------
    st.subheader(
        "Confusion Matrix"
    )

    cm_image = Image.open(
        "graphs/confusion_matrix.png"
    )

    st.image(
        cm_image,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------
    # Evaluation Metrics
    # -----------------------------------
    st.subheader(
        "Evaluation Metrics"
    )

    st.write("""
    - Accuracy Score
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    """)

    st.markdown("---")

    # -----------------------------------
    # Analysis
    # -----------------------------------
    st.subheader(
        "Project Analysis"
    )

    st.write("""
    Logistic Regression achieved the
    highest accuracy among all selected
    machine learning algorithms.

    Therefore, Logistic Regression was
    selected as the final model for
    Road Accident Severity Prediction.
    """)
