import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from wordcloud import WordCloud

# PAGE CONFIG
st.set_page_config(
    page_title="Emerging Topic Detection",
    layout="wide"
)

# TITLE
st.title("Identifying Emerging Topics in Social Media")
st.markdown(
    "Analyze social media posts using NLP and detect trending topics automatically."
)

# SIDEBAR
st.sidebar.header("Settings")

num_clusters = st.sidebar.slider(
    "Select Number of Topics",
    min_value=2,
    max_value=6,
    value=4
)

# SAMPLE POSTS
sample_posts = """
AI technology is changing the future rapidly.
The cricket match yesterday was incredible.
New smartphone launch created excitement online.
Movie trailer is trending on social media.
Election campaigns are becoming aggressive.
Football fans are celebrating the victory.
People are discussing AI tools everywhere.
The new gaming console is amazing.
Political debates are trending this week.
Social media users loved the new movie release.
"""

# INPUT
st.subheader("Input Social Media Posts")

input_method = st.radio(
    "Choose Input Method",
    ["Use Sample Data", "Paste Posts", "Upload CSV"]
)

posts = []

# SAMPLE
if input_method == "Use Sample Data":

    text = st.text_area(
        "Social Media Posts",
        sample_posts,
        height=300
    )

    posts = [
        line.strip()
        for line in text.split("\n")
        if len(line.strip()) > 5
    ]

# MANUAL INPUT
elif input_method == "Paste Posts":

    text = st.text_area(
        "Paste Social Media Posts",
        height=300
    )

    posts = [
        line.strip()
        for line in text.split("\n")
        if len(line.strip()) > 5
    ]

# CSV INPUT
else:

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_upload = pd.read_csv(uploaded_file)

        st.dataframe(df_upload.head())

        column_name = st.selectbox(
            "Select Text Column",
            df_upload.columns
        )

        posts = (
            df_upload[column_name]
            .dropna()
            .astype(str)
            .tolist()
        )

# ANALYSIS
if st.button("Analyze Topics"):

    if len(posts) < num_clusters:
        st.error("Not enough posts for clustering.")
        st.stop()

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000
    )

    X = vectorizer.fit_transform(posts)

    # K-MEANS
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    # DATAFRAME
    df = pd.DataFrame({
        "Post": posts,
        "Cluster": clusters
    })

    # TOPIC LABELS
    labels = {
        0: "Technology",
        1: "Sports",
        2: "Politics",
        3: "Entertainment",
        4: "Gaming",
        5: "Trending News"
    }

    df["Topic"] = df["Cluster"].map(labels)

    st.subheader("Detected Topics")
    st.dataframe(df)

    # BAR CHART
    topic_counts = df["Topic"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    topic_counts.plot(kind='bar', ax=ax)

    st.pyplot(fig)

    # WORD CLOUD
    all_text = " ".join(posts)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=ENGLISH_STOP_WORDS
    ).generate(all_text)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')

    st.pyplot(fig2)

    # PCA
    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X.toarray())

    pca_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Topic": df["Topic"]
    })

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x="x",
        y="y",
        hue="Topic",
        s=100
    )

    st.pyplot(fig3)

# FOOTER
st.markdown("---")
st.markdown(
    "Built using Streamlit, NLP, TF-IDF, and K-Means Clustering."
)
