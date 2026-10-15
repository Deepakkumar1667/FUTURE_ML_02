Support Ticket Classification & Prioritization
A machine-learning project that reads a customer support ticket and predicts:

Category (Ticket Type): Billing inquiry, Cancellation request, Product inquiry, Refund request, Technical issue
Priority (Ticket Priority): High, Medium, Low
Priority is predicted with an ML model and then combined with a keyword rule layer that escalates urgent tickets (e.g. "outage", "charged twice", "hacked", "asap") to High.

Project Structure
.
├── ML_project_02.ipynb          # Main notebook (full pipeline)
├── data/
│   └── customer_support_tickets.csv   # Dataset (not included, add it here)
├── outputs/                     # Generated plots and predictions
│   ├── Priority_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix_category.png
│   ├── confusion_matrix_priority.png
│   └── predictions.csv
├── requirements.txt
└── README.md
Dataset
Customer support tickets CSV (8,469 rows, 17 columns). Columns used:

Column	Use
Ticket Subject	Combined with description as model input
Ticket Description	Main text input
Ticket Type	Category target
Ticket Priority	Priority target
Pipeline
Load and clean: drop duplicates and missing values, normalise priority labels (Critical/Urgent -> High), strip {placeholder} tokens from text.
Text preprocessing: lowercase, regex tokenisation, stop-word removal (keeping negations such as "not", "cannot"), WordNet lemmatisation.
Split: 80/20 stratified train/test split (random_state=42).
Features: TF-IDF with unigrams + bigrams (min_df=2, max_features=20000, sublinear TF).
Models compared (for both tasks): Logistic Regression, Multinomial Naive Bayes, Linear SVM.
Evaluation: accuracy, weighted precision/recall/F1, classification reports, confusion matrices.
Rule-based escalation: keyword regex overrides priority to High; compared against ML only (accuracy and High-class recall).
Inference: predict_ticket(text) returns category, confidence, priority and whether a rule escalated it.
Setup
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. create the folders the notebook expects
mkdir -p data outputs
Place customer_support_tickets.csv inside data/. NLTK resources (stopwords, wordnet, omw-1.4) are downloaded automatically on first run; if offline, the notebook falls back to scikit-learn's stop-word list and skips lemmatisation.

Usage
jupyter notebook ML_project_02.ipynb
Run all cells top to bottom. Example prediction:

predict_ticket("URGENT: our whole account is down and the app is not working!")
# {'category': ..., 'confidence': ..., 'priority': 'High', 'escalated_by_rule': True}
Configuration
Set at the top of the notebook:

Variable	Default	Meaning
TEST_SIZE	0.2	Test split fraction
RANDOM_STATE	42	Reproducibility seed
USE_RULES	True	Enable keyword escalation to High
Edit HIGH_KEYWORDS to change which phrases trigger escalation.

Results (from the last notebook run)
Task	Best model	Accuracy	Weighted F1
Category	Naive Bayes	19.6%	14.5%
Priority (ML only)	Logistic Regression	45.9%	37.7%
Priority (ML + rules)	Logistic Regression + rules	46.8%	37.5%
High-priority recall improves from 82.1% (ML only) to 85.0% with the rule layer.

Known Issues / Notes
Category accuracy is at chance level (~20% for 5 classes). Part of this is a bug in the train/test split cell: x_train is assigned the category labels instead of the cleaned text. It should be:
x_train, x_test = data.loc[idx_train, "clean_text"], data.loc[idx_test, "clean_text"]
Re-run the notebook after fixing this. Note that this dataset is largely synthetic with templated descriptions, so scores may stay modest even after the fix.
Cell 7 builds needed with if c, which does not validate columns; missing_cols is computed but never checked.
The notebook saves to outputs/, so that folder must exist before running.
