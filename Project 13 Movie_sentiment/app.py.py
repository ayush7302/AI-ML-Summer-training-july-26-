import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type or paste a netflix movie review below to check if it's Positive or Negative.")

@st.cache_resource
def load_classifier():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

classifier = load_classifier()
user_review = st.text_area("Enter your movie review here:", height=150, placeholder="Type your review...")

if st.button("Analyze Sentiment"):
    if user_review.strip() != "":
        with st.spinner("Analyzing text..."):
            result = classifier(user_review)
            label = result[0]['label']
            score = result[0]['score']
            st.markdown("---")
            if label == "POSITIVE":
                st.success(f"### 🎉 Positive Review! (Confidence: {score:.2f})")
            else:
                st.error(f"### 😞 Negative Review (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text before analyzing.")
