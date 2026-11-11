# 🚐 Source Code: Cross-Faculty Van Allocation MILP Engine

โฟลเดอร์นี้รวบรวมซอร์สโค้ดและชุดข้อมูลทั้งหมดสำหรับระบบจัดสรรรถตู้โดยสารและพนักงานขับรถข้ามคณะ ด้วยตัวแบบจำลอง Mixed-Integer Linear Programming (MILP) ภายใต้เกณฑ์ความปลอดภัยชั่วโมงการทำงานสะสมไม่เกิน 60 ชั่วโมงต่อ 7 วันปฏิบัติงาน (60-Hour / 7-Day Safety Rule)

## 📁 ไฟล์สำคัญในโฟลเดอร์นี้

- **`INSTALL_GUIDE.md`**: คู่มือการติดตั้งและตั้งค่าระบบทีละขั้นตอน (Step-by-Step Installation Guide)
- **`van_allocation_engine.py`**: ตัวประมวลผลหลัก MILP Engine พร้อมโครงสร้างข้อมูลและการเรียกใช้งาน API
- **`testing_v8_limtw60.py`**: สคริปต์รันการทดลองทั้งหมด 69 Test Cases และการทดสอบ 288 Scenarios
- **`requirements.txt`**: รายการแพ็กเกจ Python ที่ต้องติดตั้ง
- **`raw_data/`**: ชุดข้อมูลประวัติการจองและข้อมูลคนขับคณะ ICT, SCI และ SEEN

## 🚀 เริ่มต้นใช้งานด่วน (Quick Start)

```bash
# 1. ติดตั้งไลบรารี
pip install -r requirements.txt

# 2. รัน Demo แก้ปัญหาจัดสรรรถตู้ (Pending Queue N=10)
python van_allocation_engine.py --mode demo

# 3. รันชุดทดสอบการทดลองฉบับเต็ม
python testing_v8_limtw60.py
```

รายละเอียดการติดตั้งและการตั้งค่าเพิ่มเติม กรุณาดูที่ [INSTALL_GUIDE.md](INSTALL_GUIDE.md)
