import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Enter a news headline or news text below.")

news = st.text_area("News")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")