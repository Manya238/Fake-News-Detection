import streamlit as st
import pickle
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

h1{
    color:#1565C0;
    text-align:center;
}

h3{
    color:#444;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    background:#1565C0;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0D47A1;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}

textarea{
    border-radius:12px;
}

</style>
""",unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model=pickle.load(open("model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history=[]

if "example" not in st.session_state:
    st.session_state.example=""

# ---------------- SIDEBAR ----------------

st.sidebar.title("📰 Fake News Detector")

st.sidebar.markdown("---")

st.sidebar.success("AI Powered Project")

st.sidebar.write("👩‍💻 Developer")
st.sidebar.info("Manya Sinha")

st.sidebar.write("🤖 Model")
st.sidebar.info("Logistic Regression")

st.sidebar.write("🎯 Accuracy")
st.sidebar.success("98.65 %")

st.sidebar.markdown("---")

st.sidebar.caption("Made using")
st.sidebar.write("✔ Python")
st.sidebar.write("✔ Streamlit")
st.sidebar.write("✔ Scikit Learn")

# ---------------- TITLE ----------------

st.title("📰 Fake News Detection System")

st.markdown(
"""
### AI Powered Fake News Detection

Check whether a News Headline or News Article is likely to be **Fake** or **Real** using Machine Learning.
"""
)

# ---------------- DASHBOARD ----------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("📄 Total News","44,898")

with c2:
    st.metric("🤖 Model","Logistic")

with c3:
    st.metric("🎯 Accuracy","98.65%")

with c4:
    st.metric("👩‍💻 Developer","Manya")

st.markdown("---")

# ---------------- EXAMPLES ----------------

st.subheader("📝 Try Example News")

a,b=st.columns(2)

with a:
    if st.button("✅ Real News Example"):
        st.session_state.example="Narendra Modi is the Prime Minister of India."

with b:
    if st.button("❌ Fake News Example"):
        st.session_state.example="Aliens have taken over the Parliament."

news=st.text_area(
    "Enter News",
    value=st.session_state.example,
    height=180
)

check=st.button("🔍 Check News")
# ---------------- PREDICTION ----------------

if check:

    if news.strip()=="":

        st.warning("⚠ Please enter some news text.")

    else:

        with st.spinner("🔍 Analyzing News..."):

            time.sleep(1.5)

            news_vector=vectorizer.transform([news])

            prediction=model.predict(news_vector)

            confidence=model.predict_proba(news_vector)

        st.markdown("---")

        if prediction[0]==0:

            result="Fake News"

            score=float(confidence[0][0])

            st.error("🚨 FAKE NEWS DETECTED")

        else:

            result="Real News"

            score=float(confidence[0][1])

            st.success("✅ REAL NEWS VERIFIED")

            st.balloons()

        st.subheader("Prediction Confidence")

        st.progress(score)

        st.write(f"### {score*100:.2f}% Confidence")

        st.session_state.history.append({

            "News":news,

            "Prediction":result,

            "Confidence":round(score*100,2)

        })

        st.markdown("---")

        st.subheader("Prediction Summary")

        x,y,z=st.columns(3)

        with x:
            st.metric("Prediction",result)

        with y:
            st.metric("Confidence",f"{score*100:.2f}%")

        with z:
            st.metric("Characters",len(news))
            # ---------------- HISTORY ----------------

st.markdown("---")

st.subheader("📜 Prediction History")

if len(st.session_state.history)==0:

    st.info("No prediction history available.")

else:

    history_df=pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv=history_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Prediction History",

        data=csv,

        file_name="prediction_history.csv",

        mime="text/csv"

    )

    if st.button("🗑 Clear History"):

        st.session_state.history=[]

        st.rerun()

    st.markdown("---")

    st.subheader("📊 Prediction Statistics")

    count=history_df["Prediction"].value_counts()

    st.bar_chart(count)

    st.subheader("🥧 Prediction Distribution")

    pie=pd.DataFrame({

        "Prediction":count.index,

        "Count":count.values

    })

    st.dataframe(
        pie,
        use_container_width=True
    )# ---------------- USER TIPS ----------------

st.markdown("---")

st.subheader("💡 Tips")

st.info("""
✔ Enter complete news headlines for better prediction.

✔ Very short text may reduce prediction accuracy.

✔ This model is trained on a news dataset and should be used for educational purposes.

✔ Always verify important news from trusted sources.
""")

# ---------------- ABOUT PROJECT ----------------

st.markdown("---")

st.subheader("📖 About this Project")

st.write("""
This Fake News Detection System uses **Machine Learning**
to classify news as **Fake** or **Real**.

### Technologies Used

- 🐍 Python
- 🎨 Streamlit
- 🤖 Scikit-learn
- 📊 Pandas
- 🔤 TF-IDF Vectorizer
- 📈 Logistic Regression

This project was developed as an AI/ML portfolio project.
""")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;">

<h3>📰 Fake News Detection System</h3>

<p>
Built with ❤️ using Python, Streamlit & Machine Learning
</p>

<p>
Developed by <b>Manya Sinha</b>
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>
""",
unsafe_allow_html=True
)