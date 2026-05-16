# ---- Step 1: Import Libraries ----
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# ---- Step 2: Load and Preprocess Dataset ----
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---- Step 3: Build Model Pipeline ----
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# ---- Step 4: Train Model ----
model.fit(df['message'], df['label'])

# ---- Step 5: Streamlit Interface ----
st.title("📧 Spam Email Classifier ")
st.write("Enter an email message below to check if it's spam or not.")

# Initialize session state for text input
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Text area for message input
msg = st.text_area("📨 Message:", key="text_input")

# Predict button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        result = model.predict([msg])[0]
        if result == 1:
            st.error("❌ This message is likely SPAM.")
        else:
            st.success("✅ This message is NOT spam.")
