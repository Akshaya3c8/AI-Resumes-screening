import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

st.set_page_config(page_title="AI Resume Screening System", layout="wide")
st.title("📄 AI Resume Screening System")
st.write("Upload candidate resumes (PDF) and match them against the Job Description.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area(
        "Enter Job Requirements/Description:",
        height=250,
        placeholder="Paste skills, qualifications, and duties here..."
    )

with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes:",
        type=["pdf"],
        accept_multiple_files=True
    )

st.markdown("---")

if st.button("🔍 Process & Rank Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        resumes_text = []
        file_names = []

        with st.spinner("Extracting text from resumes..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text.strip():
                    resumes_text.append(text)
                    file_names.append(file.name)

        if not resumes_text:
            st.error("No readable text found in the uploaded PDFs.")
        else:
            all_documents = [job_description] + resumes_text
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_documents)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            results = []
            for i in range(len(file_names)):
                match_score = round(similarity_scores[i] * 100, 2)
                results.append({
                    "Candidate Resume": file_names[i],
                    "Match Score (%)": match_score
                })
# Categorize candidates into three levels
for result in results:
    score = result["Match Score (%)"]

    if score >= 75:
        result["Level"] = "Level 1 - High Match"
    elif score >= 50:
        result["Level"] = "Level 2 - Medium Match"
    else:
        result["Level"] = "Level 3 - Low Match"

# Display results
st.success("Successfully Screened and Ranked!")
st.subheader("Candidate Ranking & Matching Score")
st.table(results)