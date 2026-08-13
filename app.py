import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="📺",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_bundle = joblib.load("linear_regression_model.pkl")

    return (
        model_bundle["model"],
        model_bundle["feature_names"]
    )


model, feature_names = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("📺 YouTube Ad Revenue Predictor")

st.markdown(
    """
    ### Predict estimated advertising revenue from video performance

    This application uses a trained **Linear Regression model** to
    estimate YouTube advertising revenue based on video performance,
    engagement, audience, device, and country information.
    """
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🎯 Enter Video Information")


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# NUMERICAL INPUTS
# ---------------------------------------------------------

with col1:

    st.subheader("📊 Video Performance")

    views = st.number_input(
        "Views",
        min_value=0,
        value=10000,
        step=100
    )

    likes = st.number_input(
        "Likes",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    comments = st.number_input(
        "Comments",
        min_value=0.0,
        value=50.0,
        step=5.0
    )

    watch_time_minutes = st.number_input(
        "Watch Time (minutes)",
        min_value=0.0,
        value=25000.0,
        step=100.0
    )


# ---------------------------------------------------------
# OTHER INPUTS
# ---------------------------------------------------------

with col2:

    st.subheader("👥 Audience & Video Details")

    video_length_minutes = st.number_input(
        "Video Length (minutes)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    subscribers = st.number_input(
        "Subscribers",
        min_value=0,
        value=50000,
        step=100
    )

    category = st.selectbox(
        "Category",
        [
            "Entertainment",
            "Gaming",
            "Education",
            "Music",
            "Tech",
            "Lifestyle"
        ]
    )

    device = st.selectbox(
        "Device",
        [
            "TV",
            "Tablet",
            "Mobile",
            "Desktop"
        ]
    )

    country = st.selectbox(
        "Country",
        [
            "IN",
            "CA",
            "UK",
            "US",
            "AU",
            "DE"
        ]
    )


# =========================================================
# ENGAGEMENT RATE
# =========================================================

if views > 0:

    engagement_rate = (likes + comments) / views

else:

    engagement_rate = 0


st.divider()

st.subheader("📈 Calculated Engagement")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:

    st.metric(
        "Engagement Rate",
        f"{engagement_rate:.4f}"
    )

with metric_col2:

    if views > 0:
        engagement_percentage = engagement_rate * 100
    else:
        engagement_percentage = 0

    st.metric(
        "Engagement %",
        f"{engagement_percentage:.2f}%"
    )


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({

    "views": [views],

    "likes": [likes],

    "comments": [comments],

    "watch_time_minutes": [watch_time_minutes],

    "video_length_minutes": [video_length_minutes],

    "subscribers": [subscribers],

    "engagement_rate": [engagement_rate]

})


# =========================================================
# CATEGORY ENCODING
# =========================================================

category_columns = [
    "Entertainment",
    "Gaming",
    "Lifestyle",
    "Music",
    "Tech"
]

for category_name in category_columns:

    input_data[f"category_{category_name}"] = (
        1 if category == category_name else 0
    )


# =========================================================
# DEVICE ENCODING
# =========================================================

device_columns = [
    "Mobile",
    "TV",
    "Tablet"
]

for device_name in device_columns:

    input_data[f"device_{device_name}"] = (
        1 if device == device_name else 0
    )


# =========================================================
# COUNTRY ENCODING
# =========================================================

country_columns = [
    "CA",
    "DE",
    "IN",
    "UK",
    "US"
]

for country_name in country_columns:

    input_data[f"country_{country_name}"] = (
        1 if country == country_name else 0
    )


# =========================================================
# ENSURE EXACT MODEL FEATURE ORDER
# =========================================================

input_data = input_data[feature_names]


# =========================================================
# PREDICTION
# =========================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Ad Revenue",
    type="primary",
    use_container_width=True
)


if predict_button:

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed successfully!")

    st.subheader("💰 Estimated Ad Revenue")

    st.metric(
        label="Predicted Revenue",
        value=f"${prediction:,.2f}"
    )

    st.caption(
        "The prediction is displayed in USD because the target "
        "variable is ad_revenue_usd."
    )


# =========================================================
# MODEL INSIGHTS
# =========================================================

st.divider()

st.header("📌 Model Insights")

insight_col1, insight_col2 = st.columns(2)


with insight_col1:

    st.subheader("🏆 Final Model")

    st.write(
        """
        **Linear Regression** was selected as the final model because
        it achieved the strongest overall performance across the
        evaluation metrics.
        """
    )

    st.write(
        """
        The model achieved an R² score of approximately **1.00**
        on the test data.
        """
    )


with insight_col2:

    st.subheader("⭐ Strongest Predictor")

    st.write(
        """
        Based on the Random Forest feature-importance analysis,
        **watch_time_minutes** was the dominant predictive feature.
        """
    )

    st.write(
        """
        This is also consistent with the strong relationship observed
        between watch time and advertising revenue during EDA.
        """
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("📊 Random Forest Feature Importance")

feature_importance_data = pd.DataFrame({

    "Feature": [
        "watch_time_minutes",
        "engagement_rate",
        "likes",
        "views",
        "subscribers",
        "video_length_minutes",
        "comments"
    ],

    "Importance": [
        0.978151,
        0.021458,
        0.000256,
        0.000099,
        0.000008,
        0.000008,
        0.000008
    ]

})

st.bar_chart(
    feature_importance_data.set_index("Feature")
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "YouTube Ad Revenue Prediction | Machine Learning Regression Project"
)