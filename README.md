# ATS-Resume-Matcher
A Flask-based NLP application that analyzes resume and job description PDFs to evaluate candidate compatibility. The system combines TF-IDF + cosine similarity with explicit skill matching to generate an ATS-style score, highlight matched and missing skills, and provide explainable hiring insights using industry-standard Post/Redirect/Get workflow.

Deployment Note
The system uses a lightweight NLP pipeline optimized for deployment on constrained environments. Heavy NLP models were intentionally avoided to ensure stability and scalability.