# Resume Analyzer & Job Match Application

## Project Description

This project is an AI-powered Resume Analyzer and Job Match application that uses **Sentence-BERT embeddings** and a **machine learning classifier** to predict how well a candidate’s resume matches a given job description. It extracts and compares skills from both documents, provides match scores, and can send email notifications with the results.

---

## Features

- Upload resume PDF and input job description text
- Extract text and skills from resume and job description using NLP
- Generate semantic embeddings using Sentence-BERT
- Predict match score using a trained classifier (Logistic Regression)
- Visualize matched and missing skills
- Optional email notifications with analysis results
- Bootstrap-based responsive frontend interface

---

## Technology Stack

- Python, Flask (backend API)
- Sentence-Transformers (BERT embeddings)
- Scikit-learn (classifier training)
- Joblib (model serialization)
- Bootstrap 5 (frontend UI)
- Flask-Mail (email notifications)

---

## Setup Instructions

### Prerequisites

- Python 3.7 or above
- Conda or virtualenv (recommended)
- Git (optional)

### Installation

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/resume-analyzer.git
cd resume-analyzer

2. Create and activate Conda environment (or use virtualenv)

conda create -n resume_match_env python=3.9 -y
conda activate resume_match_env

3. Install dependencies
pip install -r requirements.txt

4. Train the model (optional, if you want to retrain)
python resume_match_BERT.py
*** This will generate resume_matcher_bert_model.joblib in the current directory.

5. Configure email (optional)
Create a .env file with your SMTP credentials:
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_email_app_password
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
Make sure .env is in .gitignore to avoid exposing secrets.

Running the Application
1. Start the Flask backend server:
python app_bert.py

2. Open index.html in your browser (or serve it via a web server)

3. Use the form to upload a resume PDF, enter a job description, and optionally your email.

4. Submit and view the matching score, skill matches, and get notified by email if you provided one.

Project Structure
resume-analyzer/
├── app_bert.py              # Flask backend application
├── resume_match_BERT.py     # Script to train and save the ML model
├── resume_matcher_bert_model.joblib  # Saved trained model (after training)
├── bert_encoder/            # Sentence-BERT encoder files (if applicable)
├── templates/
│   └── index.html           # Frontend HTML page
├── static/                  # CSS, JS, images
├── requirements.txt         # Python dependencies
├── .env.example             # Sample environment file template
└── README.md                # Project documentation

Notes and Tips
The model is trained on a small sample dataset; for real use, gather a larger and more diverse dataset.
Always keep your .env file private and secure.
You can deploy the backend on platforms like Heroku, Render, or AWS.
For better UI/UX, customize the frontend or integrate with your existing website (e.g., Wix).

License
MIT License

Contact
For questions or contributions, please open an issue or contact [ma.abegail.mabalot@gmail.com].
