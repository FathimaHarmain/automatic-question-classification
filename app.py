
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="QClassify | Question Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# DOWNLOAD NLTK RESOURCES
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =========================================================
# NLP TOOLS
# =========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(
    "final_question_classification_model.pkl"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================
   MAIN BACKGROUND
========================= */

.stApp {
    background-color: #0b1020;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================
   REMOVE DEFAULT ELEMENTS
========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =========================
   NAVIGATION
========================= */

.nav-title {
    font-size: 21px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.nav-subtitle {
    color: #8f9ab3;
    font-size: 13px;
}


/* =========================
   HERO
========================= */

.hero-small {
    text-align: center;
    color: #8493ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 35px;
    margin-bottom: 15px;
}

.hero-title {
    text-align: center;
    color: #ffffff;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 18px;
}

.hero-highlight {
    color: #8995ff;
}

.hero-description {
    text-align: center;
    max-width: 700px;
    margin: auto;
    color: #9ca6ba;
    font-size: 16px;
    line-height: 1.7;
}


/* =========================
   SECTION HEADINGS
========================= */

.section-title {
    color: #ffffff;
    font-size: 25px;
    font-weight: 700;
    margin-top: 45px;
    margin-bottom: 5px;
}

.section-description {
    color: #8791a7;
    font-size: 14px;
    margin-bottom: 20px;
}


/* =========================
   INPUT AREA
========================= */

div[data-testid="stTextArea"] textarea {
    background-color: #11182b !important;
    color: #ffffff !important;
    border: 1px solid #28324d !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    padding: 16px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #7582ff !important;
    box-shadow: 0 0 0 1px #7582ff !important;
}


/* =========================
   BUTTON
========================= */

.stButton > button {
    background-color: #7180ff;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-size: 15px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #8290ff;
    color: white;
}


/* =========================
   RESULT
========================= */

.result-box {
    background-color: #141d34;
    border: 1px solid #34416a;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}

.result-label {
    color: #8d98b0;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-value {
    color: #9ba6ff;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}


/* =========================
   CATEGORY CARDS
========================= */

.category-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 16px;
}

.category-text {
    color: #8993a8;
    font-size: 12px;
    line-height: 1.5;
}


/* =========================
   INFO BOXES
========================= */

.info-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 15px;
}

.info-text {
    color: #8993a8;
    font-size: 12px;
    line-height: 1.6;
}


/* =========================
   FOOTER
========================= */

.footer-line {
    height: 1px;
    background-color: #222b42;
    margin-top: 60px;
    margin-bottom: 20px;
}

.footer-text {
    text-align: center;
    color: #68738a;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TOP NAVIGATION
# =========================================================

nav1, nav2 = st.columns([3, 1])

with nav1:
    st.markdown(
        '<div class="nav-title">QClassify</div>'
        '<div class="nav-subtitle">Intelligent Question Classification</div>',
        unsafe_allow_html=True
    )

with nav2:
    st.markdown(
        '<div style="text-align:right; color:#8791a7; '
        'font-size:13px; padding-top:8px;">'
        'NLP  •  ML  •  AI'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    '<div class="hero-small">AI-POWERED NLP SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'Understand Questions.<br>'
    '<span class="hero-highlight">Classify Intelligently.</span>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'An intelligent question classification system that uses '
    'Natural Language Processing and Machine Learning to '
    'automatically identify the subject of a question.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CLASSIFIER
# =========================================================

st.markdown(
    '<div class="section-title">Question Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter any question and let the trained model determine its category.'
    '</div>',
    unsafe_allow_html=True
)


question = st.text_area(
    "Question",
    placeholder=(
        "Example: What is the process by which plants "
        "convert sunlight into energy?"
    ),
    height=150,
    label_visibility="collapsed"
)


button_col1, button_col2, button_col3 = st.columns([1, 1.2, 1])

with button_col2:

    classify = st.button(
        "Classify Question",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if classify:

    if question.strip() == "":

        st.warning(
            "Please enter a question before continuing."
        )

    else:

        with st.spinner("Analyzing question..."):

            processed_question = preprocess_text(
                question
            )

            prediction = model.predict(
                [processed_question]
            )

            predicted_category = prediction[0]


        st.markdown(
            '<div class="result-box">'
            '<div class="result-label">'
            'Classification Result'
            '</div>'
            f'<div class="result-value">'
            f'{predicted_category}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# SUPPORTED CATEGORIES
# =========================================================

st.markdown(
    '<div class="section-title">Knowledge Domains</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The model classifies questions into five predefined categories.'
    '</div>',
    unsafe_allow_html=True
)


categories = [
    (
        "Science",
        "Scientific concepts, processes and phenomena."
    ),
    (
        "Mathematics",
        "Numbers, equations, calculations and mathematical concepts."
    ),
    (
        "History",
        "Historical events, people, civilizations and timelines."
    ),
    (
        "Geography",
        "Countries, locations, Earth and physical features."
    ),
    (
        "Technology",
        "Computers, software and modern technological concepts."
    )
]


cols = st.columns(5)

for col, (name, description) in zip(cols, categories):

    with col:

        with st.container(border=True):

            st.markdown(
                f'<div class="category-title">{name}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="category-text">'
                f'{description}'
                f'</div>',
                unsafe_allow_html=True
            )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The classification pipeline transforms a natural-language '
    'question into a predicted subject category.'
    '</div>',
    unsafe_allow_html=True
)


steps = [
    (
        "01",
        "Question Input",
        "The user enters a natural-language question."
    ),
    (
        "02",
        "Preprocessing",
        "The text is cleaned, tokenized, stop words are removed and words are lemmatized."
    ),
    (
        "03",
        "TF-IDF",
        "The processed question is converted into numerical text features."
    ),
    (
        "04",
        "SVM Prediction",
        "The trained Support Vector Machine predicts the category."
    )
]


cols = st.columns(4)

for col, (number, title, description) in zip(
    cols,
    steps
):

    with col:

        with st.container(border=True):

            st.markdown(
                f'<div style="color:#8290ff; font-size:12px; '
                f'font-weight:700;">STEP {number}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-title" style="margin-top:10px;">'
                f'{title}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-text" style="margin-top:8px;">'
                f'{description}'
                f'</div>',
                unsafe_allow_html=True
            )


# =========================================================
# SYSTEM INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Classifier",
        "SVM"
    )

with m2:
    st.metric(
        "Features",
        "TF-IDF"
    )

with m3:
    st.metric(
        "Dataset",
        "1,363"
    )

with m4:
    st.metric(
        "Categories",
        "5"
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">About the Project</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown(
        """
        **Automatic Question Classification System**

        This project uses Natural Language Processing and Machine
        Learning to automatically classify textual questions into
        Science, Mathematics, History, Geography and Technology.

        The system applies text preprocessing and TF-IDF feature
        extraction before using a trained Support Vector Machine
        classifier to determine the most appropriate category.

        The final model has been integrated into a Streamlit web
        application, allowing users to interact with the trained
        classification system through a simple interface.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer-line"></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-text">'
    '<b>QClassify</b> &nbsp;|&nbsp; '
    'Automatic Question Classification System'
    '<br><br>'
    'NLP • TF-IDF • Support Vector Machine • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
