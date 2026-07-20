import streamlit as st
import pickle
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

.stButton>button {
    width: 100%;
    background-color: #0066cc;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #004c99;
}

textarea {
    border-radius: 10px !important;
}

h1 {
    color: #0066cc;
}

</style>
""", unsafe_allow_html=True)

# Load Model & Vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sidebar
st.sidebar.title("📌 Project Info")
st.sidebar.write("**Project:** Fake News Detection")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Developer:** Manya Sinha")
st.sidebar.markdown("---")
st.sidebar.success("AI & Machine Learning Project")

# Main Title
st.title("📰 Fake News Detection")
st.markdown("### AI-powered Fake News Detector")
st.write("Check whether a news headline or article is likely to be **Fake** or **Real** using Machine Learning.")
st.info("Enter the news headline or news text below.")
st.info("Model Accuracy: 98.65%")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 Total News", "44,898")

with col2:
    st.metric("🎯 Accuracy", "98.65%")

with col3:
    st.metric("🤖 Model", "Logistic Regression")

with col4:
    st.metric("👩‍💻 Developer", "Manya")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "example" not in st.session_state:
    st.session_state.example = ""

# Example Buttons
st.markdown("### 📝 Try Examples")

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Real News Example"):
        st.session_state.example = "Narendra Modi is the Prime Minister of India."

with col2:
    if st.button("❌ Fake News Example"):
        st.session_state.example = "Aliens have taken over the Parliament."

# Text Box
news = st.text_area("News", value=st.session_state.example)

# Prediction
if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)
        confidence = model.predict_proba(news_vector)

        if prediction[0] == 0:
            result = "Fake News"
            conf = confidence[0][0]
            st.error("🚨 FAKE NEWS DETECTED")
        else:
            result = "Real News"
            conf = confidence[0][1]
            st.success("✅ REAL NEWS VERIFIED")
            st.balloons()

        st.progress(float(conf))
        st.write(f"### Confidence: {conf*100:.2f}%")

        st.session_state.history.append({
            "News": news[:50] + "...",
            "Result": result
        })

# Clear History
if st.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# Prediction History
st.markdown("## 📜 Prediction History")
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <h4>📰 Fake News Detection System</h4>
        <p>Built with ❤️ using Python, Streamlit & Machine Learning</p>
        <p>Developed by <b>Manya Sinha</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

if len(st.session_state.history) == 0:
    st.write("No predictions yet.")

else:
    for item in reversed(st.session_state.history):
        st.write(f"📰 **{item['News']}** → {item['Result']}")