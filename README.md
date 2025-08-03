# PrefiSence--ChatBot_Preference_Model
PrefiSence - ChatBot Preference Prediction Model
PrefiSence is a machine learning-based system designed to predict user preferences between chatbot responses. It aims to enhance chatbot evaluation by determining which response is more aligned with human expectations based on relevance, tone, and completeness.

🚀 Features
🧠 ML Model: Uses Random Forest model.

📊 Feature Engineering: Includes semantic similarity, length comparison, sentiment analysis, and more.

💬 Response Comparison Interface: Evaluate and compare chatbot responses using a custom Streamlit web app.

⚙️ Custom Dataset Support: Trained on a dataset containing human-labeled preferences between two chatbot responses per prompt.

🛠️ Tech Stack
Frontend: Streamlit

Backend: Python, Scikit-learn

Libraries:

Pandas, NumPy

Scikit-learn (Logistic Regression / Random Forest)

NLTK, SpaCy (for NLP tasks)

Sentence Transformers (BERT-based embeddings for semantic similarity)

📁 Folder Structure
bash
Copy
Edit
📦PrefiSence--ChatBot_Preference_Model
 ┣ 📂models            # Saved ML models (.pkl)
 ┣ 📂data              # Sample or training datasets
 ┣ 📂notebooks         # Jupyter notebooks for EDA, training
 ┣ 📂utils             # Helper functions for preprocessing, similarity, etc.
 ┣ app.py              # Streamlit main app
 ┣ train_model.py      # Script to train the model
 ┣ requirements.txt    # Project dependencies
 ┗ README.md           # Project overview
🧪 How It Works
Input: A prompt with two chatbot responses (A and B).

Feature Extraction: Extracts various features (semantic similarity, sentiment, etc.).

Prediction: The model predicts which response a human would prefer.

Output: Displays the preferred response.

🧑‍💻 Getting Started
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/mehaksmansoori/PrefiSence--ChatBot_Preference_Model.git
cd PrefiSence--ChatBot_Preference_Model
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
🧬 Example Use Case
Prompt: "What is the capital of France?"
Response A: "The capital of France is Paris."
Response B: "France is a beautiful country."

🧠 Prediction: PrefiSence predicts Response A as more human-preferred.

✅ Future Improvements
Integrate deep learning models like BERT for direct preference classification.

Support for multi-turn conversations.

Web deployment on Hugging Face Spaces or Streamlit Cloud.

📚 Resources & Acknowledgements
Human Preference Datasets (OpenAI, Anthropic-style)

Hugging Face Transformers & Sentence Transformers

NLTK & SpaCy NLP libraries
