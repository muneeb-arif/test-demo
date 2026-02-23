import streamlit as st
from openai import OpenAI
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------
st.set_page_config(page_title="HR AI Platform", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ HELPERS ------------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def similarity_score(text1, text2):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform([text1, text2])
    return round(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100, 2)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ------------------ SESSION STATE ------------------
if "policies" not in st.session_state:
    st.session_state.policies = ""
if "tech_questions" not in st.session_state:
    st.session_state.tech_questions = []
if "tech_answers" not in st.session_state:
    st.session_state.tech_answers = []

# ------------------ ROLE SELECTION ------------------
role = st.sidebar.selectbox("Select Role", ["HR Manager", "Employee"])
st.sidebar.write(f"Current Role: **{role}**")

st.title("🧠 HR AI Platform")

# ==================================================
# HR MANAGER VIEW
# ==================================================
if role == "HR Manager":
    tab1, tab2, tab3 = st.tabs([
        "📄 CV Evaluation",
        "📘 Policy Management",
        "🛠 Technical Evaluation"
    ])

    # -------------------------------
    # TAB 1: CV EVALUATION (MULTI CV)
    # -------------------------------
    with tab1:
        st.subheader("Candidate CV Evaluation (Multiple CVs)")
        cv_files = st.file_uploader(
            "Upload Candidate CVs (PDF/DOCX, multiple allowed)",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        jd_text = st.text_area("Paste Job Description")

        if st.button("Evaluate Candidates"):
            if cv_files and jd_text.strip():
                results = []

                with st.spinner("Evaluating CVs..."):
                    for cv in cv_files:
                        cv_text = read_pdf(cv) if cv.name.endswith(".pdf") else read_docx(cv)
                        sim_score = similarity_score(cv_text, jd_text)

                        prompt = f"""
                        You are a hiring expert.

                        Evaluate the CV and Job Description match.
                        Provide:
                        1. Eligibility percentage
                        2. Matching skills
                        3. Missing skills
                        4. Final recommendation

                        CV:
                        {cv_text}

                        Job Description:
                        {jd_text}
                        """
                        evaluation = ask_llm(prompt)
                        results.append({
                            "name": cv.name,
                            "score": sim_score,
                            "evaluation": evaluation
                        })

                # Sort by similarity score descending
                results = sorted(results, key=lambda x: x["score"], reverse=True)

                st.success("Evaluation Completed! Here are the results:")
                for i, r in enumerate(results, 1):
                    with st.expander(f"Rank {i}: {r['name']} ({r['score']}%)"):
                        st.write(r["evaluation"])
            else:
                st.warning("Upload CVs and paste Job Description")

    # -------------------------------
    # TAB 2: POLICY MANAGEMENT (MULTI PDF)
    # -------------------------------
    with tab2:
        st.subheader("Upload HR Policy PDFs (Multiple Allowed)")
        policy_files = st.file_uploader(
            "Upload policy PDFs (multiple allowed)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if policy_files:
            combined_text = ""
            for pdf in policy_files:
                combined_text += f"\n--- {pdf.name} ---\n"
                combined_text += read_pdf(pdf)
            st.session_state.policies = combined_text
            st.success(f"{len(policy_files)} policy document(s) loaded successfully")

    # -------------------------------
    # TAB 3: TECHNICAL EVALUATION
    # -------------------------------
    with tab3:
        st.subheader("Technical Assessment for Candidate")

        candidate_cv = st.file_uploader(
            "Upload Candidate CV (PDF/DOCX)",
            type=["pdf", "docx"],
            key="tech_cv"
        )
        tech_jd_text = st.text_area("Paste Job Description", key="tech_jd")

        if st.button("Generate Technical Questions"):
            if candidate_cv and tech_jd_text.strip():
                cv_text = read_pdf(candidate_cv) if candidate_cv.name.endswith(".pdf") else read_docx(candidate_cv)

                prompt = f"""
                You are a technical interviewer.

                Based on the candidate CV and the Job Description, generate up to 5 technical questions.
                Questions should increase in difficulty from low to high.
                Return questions numbered 1 to 5 in plain text.

                Candidate CV:
                {cv_text}

                Job Description:
                {tech_jd_text}
                """
                with st.spinner("Generating questions..."):
                    questions_text = ask_llm(prompt)

                # Split questions
                questions_list = [q.strip() for q in questions_text.split("\n") if q.strip()]
                st.session_state.tech_questions = questions_list
                st.session_state.tech_answers = [""] * len(questions_list)
                st.success("Questions generated! Please answer below:")

        # Display questions
        if st.session_state.tech_questions:
            answers = []
            for idx, q in enumerate(st.session_state.tech_questions):
                st.write(f"**Q{idx+1}: {q}**")
                ans = st.text_area(f"Answer for Q{idx+1}", key=f"ans_{idx}")
                answers.append(ans)

            if st.button("Submit Answers"):
                with st.spinner("Evaluating answers..."):
                    detailed_feedback = ""
                    total_score = 0
                    for i, (q, a) in enumerate(zip(st.session_state.tech_questions, answers), 1):
                        eval_prompt = f"""
                        Evaluate the candidate's answer to the following technical question.
                        Provide a score from 0 to 20 and a short feedback.

                        Question:
                        {q}

                        Candidate Answer:
                        {a}
                        """
                        result = ask_llm(eval_prompt)
                        detailed_feedback += f"**Q{i} Evaluation:**\n{result}\n\n"
                        # Optional: parse numeric score if needed
                        total_score += 0  # placeholder, can parse score later

                st.success("Technical Evaluation Completed")
                st.write(detailed_feedback)
                st.write(f"Overall Score (approximate): {total_score} / {len(st.session_state.tech_questions)*20}")

# ==================================================
# EMPLOYEE VIEW
# ==================================================
else:
    st.subheader("Ask HR Policies")

    if not st.session_state.policies:
        st.warning("HR policy documents not available. Contact HR.")
    else:
        question = st.text_input("Enter your policy question")
        if st.button("Ask"):
            if question.strip() == "":
                st.warning("Enter a question")
            else:
                prompt = f"""
                Answer ONLY using the HR policies below.
                If info not present, say "Policy does not specify this."

                POLICIES:
                {st.session_state.policies}

                QUESTION:
                {question}
                """
                with st.spinner("Searching policies..."):
                    answer = ask_llm(prompt)
                st.info("Answer")
                st.write(answer)
