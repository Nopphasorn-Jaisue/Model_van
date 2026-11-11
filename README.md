# 🚐 Van Allocation Decision Support System — 60-Hour / 7-Day Safety Rule (`Model_van`)

ระบบสนับสนุนการตัดสินใจจัดสรรรถตู้โดยสารและพนักงานขับรถข้ามคณะ ด้วยตัวแบบจำลองทางคณิตศาสตร์แบบโปรแกรมเชิงเส้นผสมจำนวนเต็ม (Mixed-Integer Linear Programming: MILP) ภายใต้เกณฑ์ความปลอดภัย **ชั่วโมงทำงานสะสมสูงสุดไม่เกิน 60 ชั่วโมงต่อ 7 วันปฏิบัติงาน (Rolling 7-day Window)** ตามมาตรฐานสากล **FMCSA (Federal Motor Carrier Safety Administration)**

---

## 📁 โครงสร้างโปรเจกต์ (Repository Structure)

```
Model_van/
├── soure_code/                         # โฟลเดอร์ซอร์สโค้ดและชุดข้อมูลพร้อมใช้งาน
│   ├── raw_data/                       # ชุดข้อมูลประวัติการจองและคนขับ (CSV)
│   │   ├── ict_booking.csv
│   │   ├── ict_drivers.csv
│   │   ├── sci_bookingN.csv
│   │   ├── sci_driversN.csv
│   │   ├── seen_booking.csv
│   │   └── seen_drivers.csv
│   ├── van_allocation_engine.py        # โมดูลตัวแก้ปัญหา MILP กลาง (Core Solver & API)
│   ├── testing_v8_limtw60.py           # สคริปต์รันการทดลอง 69 Test Cases และ 288 Scenarios
│   ├── requirements.txt                # รายการไลบรารี Python ที่ต้องติดตั้ง
│   ├── INSTALL_GUIDE.md                # คู่มือการติดตั้งและใช้งานทีละขั้นตอนอย่างละเอียด
│   └── README.md                       # สรุปภาพรวมซอร์สโค้ด
├── raw_data/                           # ชุดข้อมูลต้นฉบับ
├── INSTALL_GUIDE.md                    # คู่มือการติดตั้งและใช้งาน (Installation Manual)
├── USER_MANUAL_TH.md / .docx           # คู่มือการใช้งานระบบฉบับภาษาไทย
├── USER_MANUAL_EN.md / .docx           # คู่มือการใช้งานระบบฉบับภาษาอังกฤษ
├── report_v8.md                        # รายงานสรุปผลลัพธ์การทดลองทั้งหมด
├── EASR_manuscript_EN_revised60.docx   # ต้นฉบับบทความวิชาการภาษาอังกฤษฉบับปรับแก้เกณฑ์ 60 ชม./สัปดาห์
├── requirements.txt                    # รายการไลบรารี Python
├── testing_v8_limtw60.py               # สคริปต์หลักสำหรับประมวลผลการจัดสรรและรันการทดลองทั้งหมด
└── van_allocation_engine.py            # โมดูลตัวแก้ปัญหา MILP กลาง
```

---

## ⚙️ ข้อกำหนดความปลอดภัยที่บังคับใช้ (Safety Constraints)

1. **Daily Duty Limit:** ชั่วโมงทำงานสูงสุดไม่เกิน **14.0 ชั่วโมงต่อวันปฏิบัติงาน**
2. **Weekly Cumulative Limit:** ชั่วโมงทำงานสะสมสูงสุดไม่เกิน **60.0 ชั่วโมงต่อ 7 วันปฏิบัติงาน (Rolling 7-day Window)**
   * ป้องกันการโหมงานหนักต่อเนื่องและสอดคล้องกับเกณฑ์มาตรฐานความปลอดภัยสากล FMCSA
   * ป้องกันความเหนื่อยล้าสะสมในทริปเดินทางไกลต่อเนื่องหลายวัน
3. **Turnaround Rest Limit:** เวลาพักผ่อนขั้นต่ำระหว่างกะ/งานเดิม ไม่น้อยกว่า **10.0 ชั่วโมงต่อเนื่อง**

---

## 🚀 เริ่มต้นติดตั้งและใช้งานด่วน (Quick Start)

กรุณาศึกษาขั้นตอนอย่างละเอียดได้ที่ [INSTALL_GUIDE.md](INSTALL_GUIDE.md) หรือ [soure_code/INSTALL_GUIDE.md](soure_code/INSTALL_GUIDE.md)

### 1. ติดตั้งไลบรารี
```bash
cd soure_code
pip install -r requirements.txt
```

### 2. รันการทดสอบอย่างรวดเร็ว (Demo)
```bash
python van_allocation_engine.py --mode demo
```

### 3. รันการทดลองฉบับเต็ม (Full Experiment)
```bash
python testing_v8_limtw60.py
```
ผลลัพธ์จะถูกบันทึกและอัปเดตลงใน `report_v8.md` โดยอัตโนมัติ

---

## 📄 เอกสารอ้างอิงและการตีพิมพ์
- วิจัยและทดสอบเพื่อรองรับการตีพิมพ์บทความทางวิชาการในวารสาร Engineering and Applied Science Research (EASR)
- รายละเอียดผลการทดลอง: [report_v8.md](report_v8.md)
- คู่มือระบบ: [USER_MANUAL_TH.md](USER_MANUAL_TH.md) / [USER_MANUAL_EN.md](USER_MANUAL_EN.md)
