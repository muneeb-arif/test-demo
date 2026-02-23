from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import io
import json
import base64

app = Flask(__name__)

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# In-memory storage for session data (in production, use Redis or database)
sessions = {}

# ------------------ HELPERS ------------------
def read_pdf(file_content):
    reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file_content):
    doc = docx.Document(io.BytesIO(file_content))
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

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/evaluate-cvs', methods=['POST'])
def evaluate_cvs():
    try:
        data = request.json
        jd_text = data.get('jd_text', '')
        cv_files = data.get('cv_files', [])
        
        if not cv_files or not jd_text.strip():
            return jsonify({'error': 'Upload CVs and paste Job Description'}), 400
        
        results = []
        for cv_file in cv_files:
            cv_name = cv_file.get('name', 'Unknown')
            cv_content_base64 = cv_file.get('content', '')
            is_pdf = cv_name.lower().endswith('.pdf')
            is_docx = cv_name.lower().endswith('.docx')
            
            # Decode base64 content
            try:
                cv_content_bytes = base64.b64decode(cv_content_base64)
            except:
                return jsonify({'error': f'Error decoding file {cv_name}'}), 400
            
            # Extract text from file
            if is_pdf:
                cv_text = read_pdf(cv_content_bytes)
            elif is_docx:
                cv_text = read_docx(cv_content_bytes)
            else:
                cv_text = cv_content_bytes.decode('utf-8', errors='ignore')
            
            if not cv_text.strip():
                return jsonify({'error': f'Could not extract text from {cv_name}'}), 400
            
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
                "name": cv_name,
                "score": sim_score,
                "evaluation": evaluation
            })
        
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-policies', methods=['POST'])
def upload_policies():
    try:
        data = request.json
        policy_files = data.get('policy_files', [])
        session_id = data.get('session_id', 'default')
        
        combined_text = ""
        for pdf_file in policy_files:
            pdf_name = pdf_file.get('name', 'Unknown')
            pdf_content_base64 = pdf_file.get('content', '')
            
            # Decode base64 content
            try:
                pdf_content_bytes = base64.b64decode(pdf_content_base64)
            except:
                return jsonify({'error': f'Error decoding file {pdf_name}'}), 400
            
            # Extract text from PDF
            pdf_text = read_pdf(pdf_content_bytes)
            combined_text += f"\n--- {pdf_name} ---\n"
            combined_text += pdf_text
        
        if session_id not in sessions:
            sessions[session_id] = {}
        sessions[session_id]['policies'] = combined_text
        
        return jsonify({'success': True, 'count': len(policy_files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json
        cv_file_data = data.get('cv_file', {})
        jd_text = data.get('jd_text', '')
        session_id = data.get('session_id', 'default')
        
        if not cv_file_data or not jd_text.strip():
            return jsonify({'error': 'Upload CV and paste Job Description'}), 400
        
        cv_name = cv_file_data.get('name', 'Unknown')
        cv_content_base64 = cv_file_data.get('content', '')
        is_pdf = cv_name.lower().endswith('.pdf')
        is_docx = cv_name.lower().endswith('.docx')
        
        # Decode base64 content
        try:
            cv_content_bytes = base64.b64decode(cv_content_base64)
        except:
            return jsonify({'error': f'Error decoding file {cv_name}'}), 400
        
        # Extract text from file
        if is_pdf:
            cv_text = read_pdf(cv_content_bytes)
        elif is_docx:
            cv_text = read_docx(cv_content_bytes)
        else:
            cv_text = cv_content_bytes.decode('utf-8', errors='ignore')
        
        if not cv_text.strip():
            return jsonify({'error': f'Could not extract text from {cv_name}'}), 400
        
        prompt = f"""
        You are a technical interviewer.

        Based on the candidate CV and the Job Description, generate up to 5 technical questions.
        Questions should increase in difficulty from low to high.
        Return questions numbered 1 to 5 in plain text.

        Candidate CV:
        {cv_text}

        Job Description:
        {jd_text}
        """
        questions_text = ask_llm(prompt)
        questions_list = [q.strip() for q in questions_text.split("\n") if q.strip() and (q.strip()[0].isdigit() or q.strip().startswith('Q'))]
        
        if session_id not in sessions:
            sessions[session_id] = {}
        sessions[session_id]['tech_questions'] = questions_list
        
        return jsonify({'questions': questions_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate-answers', methods=['POST'])
def evaluate_answers():
    try:
        data = request.json
        questions = data.get('questions', [])
        answers = data.get('answers', [])
        
        if len(questions) != len(answers):
            return jsonify({'error': 'Questions and answers mismatch'}), 400
        
        detailed_feedback = ""
        for i, (q, a) in enumerate(zip(questions, answers), 1):
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
        
        return jsonify({
            'feedback': detailed_feedback,
            'total_score': f"{len(questions) * 20}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-policy', methods=['POST'])
def ask_policy():
    try:
        data = request.json
        question = data.get('question', '')
        session_id = data.get('session_id', 'default')
        
        if not question.strip():
            return jsonify({'error': 'Enter a question'}), 400
        
        policies = sessions.get(session_id, {}).get('policies', '')
        if not policies:
            return jsonify({'error': 'HR policy documents not available. Contact HR.'}), 400
        
        prompt = f"""
        Answer ONLY using the HR policies below.
        If info not present, say "Policy does not specify this."

        POLICIES:
        {policies}

        QUESTION:
        {question}
        """
        answer = ask_llm(prompt)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 HR AI Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .sidebar {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        .role-selector {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .role-selector select {
            padding: 10px 15px;
            border: 2px solid #667eea;
            border-radius: 6px;
            font-size: 16px;
            background: white;
            cursor: pointer;
        }
        .content {
            padding: 30px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        .tab {
            padding: 12px 24px;
            background: #f8f9fa;
            border: none;
            cursor: pointer;
            font-size: 16px;
            border-radius: 6px 6px 0 0;
            transition: all 0.3s;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        .form-group input[type="file"],
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            font-size: 14px;
            font-family: inherit;
        }
        .form-group textarea {
            min-height: 120px;
            resize: vertical;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 20px;
            border-radius: 6px;
        }
        .result-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        .result-item h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .alert {
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .alert-warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .question-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        .question-item label {
            font-weight: 600;
            color: #333;
            display: block;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 HR AI Platform</h1>
        </div>
        <div class="sidebar">
            <div class="role-selector">
                <label for="roleSelect">Select Role:</label>
                <select id="roleSelect">
                    <option value="hr">HR Manager</option>
                    <option value="employee">Employee</option>
                </select>
            </div>
        </div>
        <div class="content">
            <!-- HR Manager View -->
            <div id="hrView">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('cv')">📄 CV Evaluation</button>
                    <button class="tab" onclick="switchTab('policy')">📘 Policy Management</button>
                    <button class="tab" onclick="switchTab('tech')">🛠 Technical Evaluation</button>
                </div>
                
                <!-- CV Evaluation Tab -->
                <div id="cvTab" class="tab-content active">
                    <h2>Candidate CV Evaluation (Multiple CVs)</h2>
                    <div class="form-group">
                        <label>Upload Candidate CVs (PDF/DOCX, multiple allowed)</label>
                        <input type="file" id="cvFiles" multiple accept=".pdf,.docx">
                    </div>
                    <div class="form-group">
                        <label>Paste Job Description</label>
                        <textarea id="jdText" placeholder="Enter job description here..."></textarea>
                    </div>
                    <button class="btn" onclick="evaluateCVs()">Evaluate Candidates</button>
                    <div id="cvResults"></div>
                </div>
                
                <!-- Policy Management Tab -->
                <div id="policyTab" class="tab-content">
                    <h2>Upload HR Policy PDFs (Multiple Allowed)</h2>
                    <div class="form-group">
                        <label>Upload policy PDFs (multiple allowed)</label>
                        <input type="file" id="policyFiles" multiple accept=".pdf">
                    </div>
                    <button class="btn" onclick="uploadPolicies()">Upload Policies</button>
                    <div id="policyResults"></div>
                </div>
                
                <!-- Technical Evaluation Tab -->
                <div id="techTab" class="tab-content">
                    <h2>Technical Assessment for Candidate</h2>
                    <div class="form-group">
                        <label>Upload Candidate CV (PDF/DOCX)</label>
                        <input type="file" id="techCV" accept=".pdf,.docx">
                    </div>
                    <div class="form-group">
                        <label>Paste Job Description</label>
                        <textarea id="techJD" placeholder="Enter job description here..."></textarea>
                    </div>
                    <button class="btn" onclick="generateQuestions()">Generate Technical Questions</button>
                    <div id="questionsContainer"></div>
                    <div id="techResults"></div>
                </div>
            </div>
            
            <!-- Employee View -->
            <div id="employeeView" style="display: none;">
                <h2>Ask HR Policies</h2>
                <div class="form-group">
                    <label>Enter your policy question</label>
                    <input type="text" id="policyQuestion" placeholder="Enter your question here..." style="width: 100%; padding: 12px; border: 2px solid #e9ecef; border-radius: 6px; font-size: 16px;">
                </div>
                <button class="btn" onclick="askPolicy()">Ask</button>
                <div id="policyAnswer"></div>
            </div>
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Date.now();
        let currentQuestions = [];

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
        }

        function showAlert(containerId, message, type = 'success') {
            const container = document.getElementById(containerId);
            container.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
        }

        async function readFileAsBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = e => {
                    // FileReader.readAsDataURL gives us a data URL, extract base64 part
                    const dataUrl = e.target.result;
                    const base64 = dataUrl.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        async function evaluateCVs() {
            const cvFiles = document.getElementById('cvFiles').files;
            const jdText = document.getElementById('jdText').value;
            
            if (cvFiles.length === 0 || !jdText.trim()) {
                showAlert('cvResults', 'Upload CVs and paste Job Description', 'warning');
                return;
            }

            showAlert('cvResults', '<div class="spinner"></div>Evaluating CVs...', 'success');
            
            try {
                const cvFilesData = [];
                for (let file of cvFiles) {
                    const base64Content = await readFileAsBase64(file);
                    cvFilesData.push({
                        name: file.name,
                        content: base64Content
                    });
                }

                const response = await fetch('/api/evaluate-cvs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({jd_text: jdText, cv_files: cvFilesData})
                });

                const data = await response.json();
                if (data.error) {
                    showAlert('cvResults', data.error, 'error');
                } else {
                    let html = '<div class="alert alert-success">Evaluation Completed! Here are the results:</div>';
                    data.results.forEach((r, i) => {
                        html += `<div class="result-item">
                            <h4>Rank ${i+1}: ${r.name} (${r.score}%)</h4>
                            <div>${r.evaluation.replace(/\\n/g, '<br>')}</div>
                        </div>`;
                    });
                    document.getElementById('cvResults').innerHTML = html;
                }
            } catch (error) {
                showAlert('cvResults', 'Error: ' + error.message, 'error');
            }
        }

        async function uploadPolicies() {
            const policyFiles = document.getElementById('policyFiles').files;
            if (policyFiles.length === 0) {
                showAlert('policyResults', 'Please select policy files', 'warning');
                return;
            }

            showAlert('policyResults', '<div class="spinner"></div>Uploading policies...', 'success');
            
            try {
                const policyFilesData = [];
                for (let file of policyFiles) {
                    const base64Content = await readFileAsBase64(file);
                    policyFilesData.push({
                        name: file.name,
                        content: base64Content
                    });
                }

                const response = await fetch('/api/upload-policies', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({policy_files: policyFilesData, session_id: sessionId})
                });

                const data = await response.json();
                if (data.error) {
                    showAlert('policyResults', data.error, 'error');
                } else {
                    showAlert('policyResults', `${data.count} policy document(s) loaded successfully`, 'success');
                }
            } catch (error) {
                showAlert('policyResults', 'Error: ' + error.message, 'error');
            }
        }

        async function generateQuestions() {
            const techCV = document.getElementById('techCV').files[0];
            const techJD = document.getElementById('techJD').value;
            
            if (!techCV || !techJD.trim()) {
                showAlert('techResults', 'Upload CV and paste Job Description', 'warning');
                return;
            }

            showAlert('techResults', '<div class="spinner"></div>Generating questions...', 'success');
            
            try {
                const cvBase64Content = await readFileAsBase64(techCV);
                const response = await fetch('/api/generate-questions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        cv_file: {
                            name: techCV.name,
                            content: cvBase64Content
                        },
                        jd_text: techJD,
                        session_id: sessionId
                    })
                });

                const data = await response.json();
                if (data.error) {
                    showAlert('techResults', data.error, 'error');
                } else {
                    currentQuestions = data.questions;
                    let html = '<div class="alert alert-success">Questions generated! Please answer below:</div>';
                    data.questions.forEach((q, idx) => {
                        html += `<div class="question-item">
                            <label>Q${idx+1}: ${q}</label>
                            <textarea id="answer_${idx}" style="width: 100%; min-height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                        </div>`;
                    });
                    html += '<button class="btn" onclick="submitAnswers()" style="margin-top: 20px;">Submit Answers</button>';
                    document.getElementById('questionsContainer').innerHTML = html;
                }
            } catch (error) {
                showAlert('techResults', 'Error: ' + error.message, 'error');
            }
        }

        async function submitAnswers() {
            const answers = currentQuestions.map((_, idx) => {
                return document.getElementById(`answer_${idx}`).value;
            });

            showAlert('techResults', '<div class="spinner"></div>Evaluating answers...', 'success');
            
            try {
                const response = await fetch('/api/evaluate-answers', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({questions: currentQuestions, answers: answers})
                });

                const data = await response.json();
                if (data.error) {
                    showAlert('techResults', data.error, 'error');
                } else {
                    let html = '<div class="alert alert-success">Technical Evaluation Completed</div>';
                    html += `<div class="result-box">${data.feedback.replace(/\\n/g, '<br>')}</div>`;
                    html += `<div style="margin-top: 15px;"><strong>Overall Score (approximate):</strong> ${data.total_score}</div>`;
                    document.getElementById('techResults').innerHTML = html;
                }
            } catch (error) {
                showAlert('techResults', 'Error: ' + error.message, 'error');
            }
        }

        async function askPolicy() {
            const question = document.getElementById('policyQuestion').value;
            if (!question.trim()) {
                showAlert('policyAnswer', 'Enter a question', 'warning');
                return;
            }

            showAlert('policyAnswer', '<div class="spinner"></div>Searching policies...', 'success');
            
            try {
                const response = await fetch('/api/ask-policy', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question, session_id: sessionId})
                });

                const data = await response.json();
                if (data.error) {
                    showAlert('policyAnswer', data.error, 'error');
                } else {
                    document.getElementById('policyAnswer').innerHTML = 
                        `<div class="alert alert-success"><strong>Answer:</strong><br>${data.answer.replace(/\\n/g, '<br>')}</div>`;
                }
            } catch (error) {
                showAlert('policyAnswer', 'Error: ' + error.message, 'error');
            }
        }

        document.getElementById('roleSelect').addEventListener('change', function() {
            const role = this.value;
            if (role === 'hr') {
                document.getElementById('hrView').style.display = 'block';
                document.getElementById('employeeView').style.display = 'none';
            } else {
                document.getElementById('hrView').style.display = 'none';
                document.getElementById('employeeView').style.display = 'block';
            }
        });
    </script>
</body>
</html>'''

if __name__ == '__main__':
    app.run(debug=True)
