# Automatic Question Classification System

## Project Overview
The Automatic Question Classification System is an NLP-based machine learning application that automatically classifies textual questions into five categories: Science, Mathematics, History, Geography, and Technology.

## Objectives
- Automate the classification of educational questions.
- Apply NLP preprocessing and TF-IDF feature extraction.
- Use machine learning to predict question categories.
- Provide a simple, user-friendly web interface.

## Technologies Used
- Python
- Natural Language Processing (NLTK)
- TF-IDF Vectorization
- Support Vector Machine (LinearSVC)
- Scikit-learn
- Streamlit

## Model Performance
The selected SVM model achieved **83.15% accuracy** on the project's test dataset.

## Application Features
- Text input for questions
- Automatic subject classification
- Web-based user interface
- Predictions across five subject categories

## How to Run Locally
1. Install the dependencies using `pip install -r requirements.txt`.
2. Run the application using `streamlit run app.py`.
3. Open the local URL displayed in the terminal.

## Future Enhancements
- Improve classification of underrepresented categories.
- Expand the training dataset.
- Evaluate additional NLP models.
- Improve prediction reliability for ambiguous questions.
