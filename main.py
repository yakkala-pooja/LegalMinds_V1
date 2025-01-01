import os
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'E:/Copies'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Load the Legal-BERT model and tokenizer
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def get_content(filename):
    file_path = os.path.join("E:/Copies", filename)
    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read()
    return content

def search_files(query):
    files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if query.lower() in filename.lower():
            files.append(filename)
    return files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/legal_user')
def legal_user():
    return render_template('legal_user.html')

@app.route('/document_analysis', methods=['GET', 'POST'])
def document_analysis():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return redirect(request.url)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
        return redirect(url_for('document_analysis'))
    if request.method == 'GET':
        search_query = request.args.get('search', '')
        if search_query:
            files = search_files(search_query)
            return render_template('document_analysis.html', files=files, search_query=search_query)
    return render_template('document_analysis.html')

@app.route("/open_file/<filename>")
def get_file(filename):
    if filename.endswith(".txt"):
        file_contents = get_content(filename)
        return render_template("file_display.html", filename=filename, file_contents=file_contents)
    else:
        return render_template("file_display.html", filename=filename)

from summary_gen import generate_summary

@app.route("/summary/<filename>")
def get_summary(filename):
    if filename.endswith(".txt"):
        file_contents = get_content(filename)
        summary = generate_summary(file_contents)
        return render_template("summary.html", filename=filename, summary=summary)
    else:
        return render_template("summary.html", filename=filename)

import spacy

nlp = spacy.load("en_core_web_sm")

@app.route("/informatics/<filename>")
def get_informatics(filename):
    document_content = get_content(filename)
    doc = nlp(document_content)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return render_template("informatics.html", filename=filename, entities=entities)

# Route for chatbot page
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

# Route for chatbot responses
@app.route('/chatbot-response', methods=['POST'])
def chatbot_response():
    # Extract user query from POST request
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({"response": "Please provide a valid legal query."})

    # Load legal context from files in UPLOAD_FOLDER
    context = ""
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(UPLOAD_FOLDER, filename), 'r', encoding='utf-8') as file:
                context += file.read() + "\n"

    if not context:
        return jsonify({"response": "No legal context available to answer your query."})

    # Generate an answer from the context
    try:
        result = qa_pipeline(question=user_query, context=context)
        answer = result['answer']
    except Exception as e:
        answer = "Sorry, I could not process your query. Please refine your question."

    return jsonify({"response": answer})


if __name__ == '__main__':
    app.run(debug=True)