from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pythainlp.tokenize import sent_tokenize, word_tokenize
import torch
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# โหลดโมเดลสำหรับ Abstractive Summarization
abstractive_model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(abstractive_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(abstractive_model_name)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    import fitz  # pymupdf
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extractive_summarization(text, length_option):
    # แบ่งข้อความเป็นประโยค
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "ไม่สามารถสรุปเนื้อหาได้"

    # สร้างเวกเตอร์ TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # คำนวณความคล้ายคลึงกันของประโยค
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    # สร้างกราฟ
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # จัดอันดับประโยคตามคะแนน
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # กำหนดความยาวสรุป
    if length_option == 'short':
        num_sentences = max(1, int(len(sentences) * 0.2))
    elif length_option == 'medium':
        num_sentences = max(1, int(len(sentences) * 0.5))
    else:  # long
        num_sentences = max(1, int(len(sentences) * 0.8))

    # เลือกประโยคที่สำคัญที่สุด
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)
    return summary

def abstractive_summarization(text, length_option):
    # กำหนดความยาวสรุป
    if length_option == 'short':
        max_length = 50
        min_length = 25
    elif length_option == 'medium':
        max_length = 100
        min_length = 50
    else:  # long
        max_length = 200
        min_length = 100

    # เตรียมข้อมูลสำหรับโมเดล
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True)

    # ตรวจสอบว่ามีการใช้ GPU หรือไม่
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = input_ids.to(device)

    # สร้างสรุป
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # ดึงข้อความจากไฟล์ PDF
                text = extract_text_from_pdf(filepath)
                # ลบไฟล์ PDF หลังจากใช้งานเสร็จ
                os.remove(filepath)
            else:
                return "ไฟล์ไม่ถูกต้องหรือไม่ได้รับอนุญาต"
        else:
            text = request.form['text']

        method = request.form['method']
        length = request.form['length']

        if method == 'extractive':
            summary = extractive_summarization(text, length)
        else:
            summary = abstractive_summarization(text, length)

        return render_template('result.html', summary=summary, original_text=text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
