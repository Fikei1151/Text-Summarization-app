from flask import Flask, render_template, request
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from torch import cuda

app = Flask(__name__)

# ตรวจสอบว่า GPU ใช้งานได้หรือไม่
device = "cuda" if cuda.is_available() else "cpu"

# โหลด tokenizer และโมเดลจาก Hugging Face โดยตรง
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = MT5ForConditionalGeneration.from_pretrained("csebuetnlp/mT5_multilingual_XLSum").to(device)

# ฟังก์ชันสำหรับสรุปข้อความ
def summarize_text(sentence, length_type="short"):
    # ตั้งค่าพารามิเตอร์สำหรับการสรุป
    if length_type == "short":
        min_length = 10
        max_length = 50
    elif length_type == "medium":
        min_length = 50
        max_length = 100
    elif length_type == "long":
        min_length = 100
        max_length = 150
    else:
        raise ValueError("Invalid length_type. Choose from 'short', 'medium', or 'long'.")

    # แปลงข้อความให้เป็น input tensor
    inputs = tokenizer(sentence, return_tensors="pt").input_ids.to(device)

    # ทำการทำนายผล โดยกำหนดพารามิเตอร์เพิ่มเติมเพื่อลดความซ้ำซ้อน
    outputs = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=3,               # ลดจำนวน beams เพื่อลดความซ้ำซ้อน
        early_stopping=True,        # หยุดเมื่อโมเดลคิดว่าได้คำตอบที่สมเหตุสมผล
        no_repeat_ngram_size=3,     # ป้องกันการซ้ำของ n-grams
        length_penalty=1.5,         # ปรับให้ผลลัพธ์มีความยาวสมเหตุสมผล
        repetition_penalty=2.5,     # เพิ่มบทลงโทษสำหรับการทำนายคำซ้ำ
        top_p=0.92,                 # ใช้ nucleus sampling เพื่อเลือกคำที่ดีที่สุด
        top_k=50,                   # จำกัดการเลือกคำที่อยู่ใน top 50
        temperature=0.7             # ลดความแปรปรวนของการทำนาย
    )

    # ถอดรหัสผลลัพธ์ที่ได้
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    
    return prediction

# หน้าแรกที่มีฟอร์มให้ผู้ใช้กรอกข้อความ
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # รับค่าจากฟอร์ม
        text = request.form["text"]
        summary_type = request.form["summary_type"]
        
        # สรุปข้อความ
        summary = summarize_text(text, summary_type)
        
        return render_template("index.html", summary=summary, text=text, summary_type=summary_type)

    return render_template("index.html", summary=None)

if __name__ == "__main__":
    app.run(debug=True,port=5001)
