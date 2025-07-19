# resume_match_bert.py

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example dataset
resumes = [
    "Experienced software engineer with 5 years in Python, Java, and cloud platforms like AWS and Azure. Worked on backend systems, REST APIs, and DevOps tools.",
    "Marketing professional with strong skills in SEO, social media campaigns, and Google Ads. Managed digital marketing budgets and brand awareness projects.",
    "Data scientist with experience in machine learning, deep learning, and NLP. Proficient in Python, TensorFlow, and building predictive models.",
    "Customer service representative with excellent communication and problem-solving skills. Experience handling customer complaints and CRM systems.",
    "Graphic designer skilled in Adobe Photoshop, Illustrator, and UI/UX design principles. Created branding materials and wireframes.",
    "Junior accountant with skills in bookkeeping, QuickBooks, and tax preparation.",
    "Sales executive with experience in B2B sales, lead generation, and CRM platforms.",
    "Construction manager experienced in on-site safety, logistics, and civil engineering.",
    "Teacher with 10 years experience in curriculum development and online teaching platforms.",
    "Nurse with experience in ER, patient monitoring, and healthcare reporting."
]

jobs = [
    "Looking for a backend software engineer skilled in Python, cloud computing (AWS), and RESTful API development.",
    "Seeking a digital marketing specialist with expertise in SEO, paid search, and content strategy for online campaigns.",
    "Hiring data scientist with knowledge of machine learning, Python, and NLP to work on AI projects.",
    "Customer support associate needed to resolve tickets, communicate with clients, and use CRM software effectively.",
    "Opening for a creative designer with experience in branding, Adobe Creative Suite, and user interface design.",
    "Looking for a backend software engineer skilled in Python, cloud computing (AWS), and RESTful API development.",
    "Seeking a digital marketing specialist with expertise in SEO, paid search, and content strategy for online campaigns.",
    "Hiring data scientist with knowledge of machine learning, Python, and NLP to work on AI projects.",
    "Customer support associate needed to resolve tickets, communicate with clients, and use CRM software effectively.",
    "Opening for a creative designer with experience in branding, Adobe Creative Suite, and user interface design."
]

# Labels: 1 = good match, 0 = mismatch
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Create pair embeddings using absolute difference of BERT vectors
resume_embeddings = model.encode(resumes)
job_embeddings = model.encode(jobs)

features = np.abs(resume_embeddings - job_embeddings)

# Train a simple classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and BERT encoder
joblib.dump(clf, 'resume_matcher_bert_model.joblib')
model.save('bert_encoder')
