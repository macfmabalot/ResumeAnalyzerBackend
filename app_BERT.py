# app_bert.py

from flask import Flask, request, jsonify
import joblib
from sentence_transformers import SentenceTransformer
import os
import docx, PyPDF2
import spacy
nlp = spacy.load("en_core_web_sm")
from flask_mail import Mail, Message


app = Flask(__name__)
clf = joblib.load('resume_matcher_bert_model.joblib')
encoder = SentenceTransformer('bert_encoder')

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='ma.abegail.mabalot@gmail.com',
    MAIL_PASSWORD=''
)

mail = Mail(app)


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def extract_skills(text):
    doc = nlp(text.lower())
    skill_phrases = []
    
    # Example: you can expand this set
    known_skills = {
        'python', 'java', 'c++', 'sql', 'aws', 'azure', 'excel', 'power bi',
        'machine learning', 'deep learning', 'data analysis', 'nlp', 'seo',
        'tensorflow', 'keras', 'scikit-learn', 'docker', 'react', 'node.js',
        'pytorch', 'git', 'adobe photoshop', 'illustrator', 'crm', 'google ads'
    }

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        if cleaned in known_skills:
            skill_phrases.append(cleaned)

    # remove duplicates
    return list(set(skill_phrases))

@app.route('/analyze', methods=['POST'])
def analyze():
    resume = request.files['resume']
    jobdesc = request.form['jobdesc']
    user_email = request.form.get('email')  # get email from form

    path = os.path.join('uploads', resume.filename)
    resume.save(path)
    resume_text = extract_text(path)

    res_embed = encoder.encode([resume_text])[0]
    job_embed = encoder.encode([jobdesc])[0]
    feature = abs(res_embed - job_embed).reshape(1, -1)
    score = clf.predict_proba(feature)[0][1]

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(jobdesc)

    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]

    # Compose email body
    body = f"""
    Resume Analysis Results:

    Match Score: {round(score * 100, 2)}%

    Matched Skills:
    {', '.join(matched) if matched else 'None'}

    Missing Skills:
    {', '.join(missing) if missing else 'None'}

    Thank you for using our Resume Analyzer.
    """

    # Send email if user provided email
    if user_email:
        msg = Message(subject="Your Resume Analysis Results",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[user_email],
                      body=body)


    return jsonify({
        'match_score': round(float(score) * 100, 2),
        'skills': {
            'matched': matched,
            'missing': missing
        }
    })




if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
