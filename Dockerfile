# ใช้ base image ของ Python 3.10-slim
FROM python:3.10-slim

# ตั้งค่าตัวแปรสำหรับไม่ต้องการ interactive
ENV DEBIAN_FRONTEND=noninteractive

# ติดตั้ง dependencies ที่จำเป็น
RUN apt-get update && \
    apt-get install -y git build-essential libgl1-mesa-glx libglib2.0-0 \
    python3-dev libicu-dev libthai-dev unzip wget && \
    rm -rf /var/lib/apt/lists/*

# สร้างไดเรกทอรีสำหรับแอป
WORKDIR /app

# สร้างโฟลเดอร์สำหรับเก็บไฟล์ที่อัปโหลด
RUN mkdir uploads

# คัดลอกไฟล์ requirements.txt และติดตั้งไลบรารี Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดไปยังไดเรกทอรี /app
COPY . /app

# เปิดพอร์ต 8000
EXPOSE 8000

# รันแอปพลิเคชัน
CMD ["python", "app.py"]
