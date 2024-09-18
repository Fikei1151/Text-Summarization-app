from transformers import MT5ForConditionalGeneration, AutoTokenizer
from torch import cuda

# ตรวจสอบว่า GPU ใช้งานได้หรือไม่
device = "cuda" if cuda.is_available() else "cpu"

# โหลด tokenizer และโมเดลจาก Hugging Face โดยตรง
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = MT5ForConditionalGeneration.from_pretrained("csebuetnlp/mT5_multilingual_XLSum").to(device)
#tokenizer = AutoTokenizer.from_pretrained("thanathorn/mt5-cpe-kmutt-thai-sentence-sum")
#model = MT5ForConditionalGeneration.from_pretrained("thanathorn/mt5-cpe-kmutt-thai-sentence-sum").to(device)

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
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return prediction

# ข้อความที่ต้องการสรุป
sentence = """ผู้ต้องสงสัยคดีพยายามลอบสังหาร โดนัลด์ ทรัมป์ ครั้งที่ 2 ถูกตั้งข้อหาเกี่ยวกับอาวุธปืน 2 กระทง และมีโอกาสที่เขาจะถูกตั้งข้อหาเพิ่มเติมอีก

เมื่อวันจันทร์ที่ 16 ก.ย. 2567 นาย ไรอัน เวสลีย์ เราท์ ผู้ต้องสงสัยในคดีที่ดูเหมือนเป็นความพยายามลอบสังหาร โดนัลด์ ทรัมป์ อดีตประธานาธิบดีสหรัฐฯ เป็นครั้งที่ 2 ในรอบเกือบ 2 เดือน ถูกตั้งข้อหาเกี่ยวกับอาวุธปืน 2 กระทงได้แก่ ครอบครองอาวุธปืนในขณะที่เป็นผู้ถูกตัดสินว่ามีความผิดในคดีอาญา และครอบครองปืนที่เลขทะเบียนถูกทำลาย

อย่างไรก็ตาม นายเราท์ วัย 58 ปี อาจถูกตั้งข้อหาเพิ่มเติมอีก เมื่อการสืบสวนมีความคืบหน้า โดยข้อกล่าวหาเกี่ยวกับอาวุธปืนดังกล่าว เป็นข้อหาที่อัยการยื่นต่อศาลเพื่อให้สามารถคุมขังนายเราท์ได้ต่อไป ระหว่างเจ้าหน้าที่สืบสวนข้อเท็จจริง โดยนายเราท์มีกำหนดการขึ้นศาลเรื่องการควบคุมตัวในวันจันทร์หน้า (23 ก.ย.)

ทั้งนี้ จากการสืบสวนเบื้องต้นพบว่า นายเราท์ เป็นเจ้าของบริษัทก่อสร้างเล็กๆ แห่งหนึ่งในรัฐฮาวาย เขาวางแผนซุ่มยิง โดนัลด์ ทรัมป์ ตอนที่เขาออกรอบตีกอล์ฟที่ “ทรัมป์ อินเทอร์เนชันแนล กอล์ฟ คลับ” ในเมืองเวสต์ ปาล์ม บีช รัฐฟลอริดา วันอาทิตย์ที่ 15 ก.ย. 2567 ที่ผ่านมา

แต่หน่วยตำรวจลับสังเกตเห็นปากกระบอกปืนไรเฟิลของนายเราท์ ยื่นออกมาจากรั้วซึ่งเต็มไปด้วยพุ่มไม้ จึงยิงสกัดเอาไว้ ทำให้นายเราท์วิ่งกลับไปขึ้นรถแล้วหลบหนีไป ก่อนจะถูกจับตัวได้กลางถนนหลวงในเขตข้างเคียงในเวลาต่อมา โดยที่นายทรัมป์ไม่ได้รับบาดเจ็บใดๆ ในเหตุการณ์นี้

เจ้าหน้าที่ระบุว่า นายเราท์เคยโพสต์ข้อความวิพากษ์วิจารณ์นายทรัมป์บนโลกออนไลน์มาก่อน และเป็นผู้สนับสนุนการช่วยเหลือยูเครนรับมือกับการรุกรานของรัสเซียตัวยง ก่อนเกิดเหตุเขามีรายได้เดือนละ 3,000 ดอลลาร์สหรัฐ และไม่มีทรัพย์สินอื่นนอกจากรถกระบะ 2 คันที่ฮาวาย"""

# สรุปแบบสั้น
print("สรุปแบบสั้น:", summarize_text(sentence, "short"))

# สรุปแบบกลาง
print("สรุปแบบกลาง:", summarize_text(sentence, "medium"))

# สรุปแบบยาว
print("สรุปแบบยาว:", summarize_text(sentence, "long"))
