# app_bert.py

import joblib
from sentence_transformers import SentenceTransformer
import os
import docx, PyPDF2
import spacy
nlp = spacy.load("en_core_web_sm")
from flask_mail import Mail, Message
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Optional: allows frontend requests from a different origin

clf = joblib.load('resume_matcher_bert_model.joblib')
encoder = SentenceTransformer('bert_encoder')

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
        'pytorch', 'git', 'adobe photoshop', 'illustrator', 'crm', 'google ads',
        'HTML', 'CSS','ASP.net', 'Visual Basics', 'MVC C#'
    }

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        if cleaned in known_skills:
            skill_phrases.append(cleaned)

    # remove duplicates
    return list(set(skill_phrases))

@app.route('/analyze', methods=['POST'])
def analyze():
    jobdesc = request.form.get("jobdesc")
    resume = request.files.get("resume")

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

    if not jobdesc or not resume:
        return jsonify({"error": "Missing job description or resume"}), 400

    # 🧪 For now: just return dummy match score and skills
    return jsonify({
        'match_score': round(float(score) * 100, 2),
        'skills': {
            'matched': matched,
            'missing': missing
        }
    })

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
