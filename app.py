from flask import *
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

# ---------------- CONFIG ---------------- #

SKILL_KEYWORDS = {
    "python", "flask", "django", "react", "node", "express", "mern",
    "javascript", "html", "css", "sql", "mongodb", "mysql",
    "machine learning", "deep learning", "nlp", "data science",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "aws", "api", "rest",
    "java", "c++", "c", "linux"
}

nlp = spacy.load("en_core_web_lg")
app = Flask(__name__)

# ---------------- UTILS ---------------- #

def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def clean_fun(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|\d+", " ", text)

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)


def extract_skills(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|\d+", " ", text)

    doc = nlp(text)
    found = set()

    for chunk in doc.noun_chunks:
        skill = chunk.text.strip()
        if skill in SKILL_KEYWORDS:
            found.add(skill)

    for token in doc:
        if token.text in SKILL_KEYWORDS:
            found.add(token.text)

    return sorted(found)

# ---------------- ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        resume_pdf = request.files.get("resume_pdf")
        jd_pdf = request.files.get("jd_pdf")

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        clean_resume = clean_fun(resume_text)
        clean_jd = clean_fun(jd_text)

        # Semantic similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([clean_resume, clean_jd])
        semantic_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
        semantic_score = round(semantic_score, 2)

        # Skill extraction
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        matched_skills = sorted(set(resume_skills) & set(jd_skills))
        missing_skills = sorted(set(jd_skills) - set(resume_skills))

        skill_match_pct = (
            (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
        )
        skill_match_pct = round(skill_match_pct, 2)

        # Final ATS score
        final_score = round((0.6 * semantic_score) + (0.4 * skill_match_pct), 2)

        # Classification
        if final_score < 30:
            level = "❌ Bad Match"
        elif final_score < 60:
            level = "⚠️ Average Match"
        elif final_score < 80:
            level = "✅ Good Match"
        else:
            level = "🔥 Strong Match"

        return render_template(
            "home.html",
            semantic_score=semantic_score,
            skill_match_pct=skill_match_pct,
            final_score=final_score,
            level=level,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
