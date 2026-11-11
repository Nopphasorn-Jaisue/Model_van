# 📖 คู่มือการติดตั้งและใช้งานซอร์สโค้ด (Installation & User Guide)
## ระบบจัดสรรรถตู้โดยสารส่วนกลางข้ามคณะด้วยโมเดล MILP (Van Allocation DSS — 60-Hour Rule)

---

## 📌 สารบัญ (Table of Contents)
1. [ภาพรวมของระบบ (System Overview)](#1-ภาพรวมของระบบ-system-overview)
2. [โครงสร้างไฟล์ในโฟลเดอร์ soure_code](#2-โครงสร้างไฟล์ในโฟลเดอร์-soure_code)
3. [ความต้องการของระบบ (System Requirements)](#3-ความต้องการของระบบ-system-requirements)
4. [ขั้นตอนการติดตั้งทีละขั้นตอน (Step-by-Step Installation)](#4-ขั้นตอนการติดตั้งทีละขั้นตอน-step-by-step-installation)
5. [การทดสอบและรันโปรแกรม (Execution & Verification)](#5-การทดสอบและรันโปรแกรม-execution--verification)
6. [ข้อกำหนดรูปแบบข้อมูลนำเข้า (Input Data Specification)](#6-ข้อกำหนดรูปแบบข้อมูลนำเข้า-input-data-specification)
7. [กฎความปลอดภัย Hours-of-Service (HOS Safety Rules)](#7-กฎความปลอดภัย-hours-of-service-hos-safety-rules)
8. [การแก้ปัญหาที่พบบ่อย (Troubleshooting & FAQ)](#8-การแก้ปัญหาที่พบบ่อย-troubleshooting--faq)

---

## 1. ภาพรวมของระบบ (System Overview)

ชุดซอร์สโค้ดนี้เป็น **ระบบสนับสนุนการตัดสินใจ (Decision Support System: DSS)** ที่พัฒนาขึ้นด้วยการจำลองทางคณิตศาสตร์แบบ **Mixed-Integer Linear Programming (MILP)** เพื่อจัดสรรรถตู้โดยสารและคนขับร่วมกันข้ามคณะ (Cross-Faculty Shared Fleet) 

### จุดเด่นของระบบ:
- **Lexicographic MILP Optimization:** ลำดับความสำคัญวัตถุประสงค์ (1) รับส่งผู้โดยสารให้ได้มากที่สุด (2) ตอบสนองคำขอครบถ้วน (3) ลดการยืมรถข้ามคณะโดยไม่จำเป็น (4) ลดภาระงานล่วงเวลาของพนักงานขับรถ
- **กฎความปลอดภัยชั่วโมงการทำงาน (Hours-of-Service - HOS):** สอดคล้องตามมาตรฐาน FMCSA และกฎหมายไทย โดยใช้เกณฑ์ความปลอดภัย **ชั่วโมงทำงานสะสมไม่เกิน 60 ชั่วโมงต่อ 7 วันปฏิบัติงาน (Rolling 7-day Window)**, พักต่อเนื่องขั้นต่ำ 10 ชั่วโมง และทำงานสูงสุดไม่เกิน 14 ชั่วโมง/วัน
- **ตัวแก้ปัญหาโอเพนซอร์ส:** ใช้งานผ่านไลบรารี `PuLP` ร่วมกับ **COIN-OR CBC Solver** ที่ติดตั้งมาในตัว โดยไม่ต้องเสียค่าลิขสิทธิ์ซอฟต์แวร์พาณิชย์

---

## 2. โครงสร้างไฟล์ในโฟลเดอร์ `soure_code`

```
soure_code/
├── raw_data/                           # โฟลเดอร์ชุดข้อมูลนำเข้า (.csv)
│   ├── ict_booking.csv                 # ข้อมูลการจองรถของคณะ ICT
│   ├── ict_drivers.csv                 # ข้อมูลการทำงานของคนขับคณะ ICT
│   ├── sci_bookingN.csv                # ข้อมูลการจองรถจริงของคณะ SCI
│   ├── sci_driversN.csv                # ข้อมูลคนขับจริงของคณะ SCI
│   ├── seen_booking.csv                # ข้อมูลการจองรถกรณี Stress-test คณะ SEEN
│   └── seen_drivers.csv                # ข้อมูลคนขับกรณี Stress-test คณะ SEEN
├── van_allocation_engine.py            # โมดูลตัวแก้ปัญหา MILP กลาง (Core Engine & API)
├── testing_v8_limtw60.py               # สคริปต์รันชุดทดลอง 69 Test Cases และ 288 Scenarios
├── requirements.txt                    # รายการไลบรารี Python ที่จำเป็น
├── INSTALL_GUIDE.md                    # คู่มือการติดตั้งและใช้งานฉบับนี้
└── README.md                           # สรุปภาพรวมของซอร์สโค้ด
```

---

## 3. ความต้องการของระบบ (System Requirements)

| รายการ | ความต้องการขั้นต่ำ | แนะนำสำหรับการทดลองขนาดใหญ่ |
| :--- | :--- | :--- |
| **ระบบปฏิบัติการ** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11 / Ubuntu 22.04 LTS |
| **Python** | Python 3.9 - 3.12 | Python 3.10 หรือ 3.11 |
| **หน่วยประมวลผล (CPU)** | 2 Cores | 4 Cores ขึ้นไป |
| **หน่วยความจำ (RAM)** | 4 GB | 8 GB ขึ้นไป |
| **พื้นที่ว่างในดิสก์** | 500 MB | 1 GB |
| **Solver** | COIN-OR CBC (มาพร้อมกับ `pulp`) | COIN-OR CBC / GLPK |

---

## 4. ขั้นตอนการติดตั้งทีละขั้นตอน (Step-by-Step Installation)

### 4.1 ตรวจสอบและติดตั้ง Python
ตรวจสอบว่าในเครื่องมี Python ติดตั้งอยู่แล้วหรือไม่ โดยเปิด Terminal (macOS/Linux) หรือ PowerShell (Windows):

```bash
python --version
# หรือ
python3 --version
```
> *หมายเหตุ:* หากยังไม่มี ให้ดาวน์โหลดและติดตั้งจาก [python.org](https://www.python.org/downloads/) (อย่าลืมทำเครื่องหมายถูกที่ช่อง **"Add Python to PATH"**)

---

### 4.2 เปิดโฟลเดอร์ซอร์สโค้ด
```bash
cd soure_code
```

---

### 4.3 สร้างและเปิดใช้งาน Virtual Environment (แนะนำอย่างยิ่ง)

#### บน Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(หากพบข้อผิดพลาดเรื่อง Execution Policy บน PowerShell ให้รัน `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ก่อน)*

#### บน Windows (Command Prompt - cmd):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### บน macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4.4 ติดตั้งไลบรารีที่จำเป็น (Dependencies)

รันคำสั่งติดตั้งแพ็กเกจจาก `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### รายการแพ็กเกจหลัก:
- `pulp>=2.7.0`: ไลบรารี MILP Modeling และตัวเชื่อมต่อ CBC Solver
- `pandas>=2.0.0`: จัดการโครงสร้างข้อมูลตาราง
- `numpy>=1.24.0`: การคำนวณทางคณิตศาสตร์และเมทริกซ์
- `matplotlib>=3.7.0`: สร้างแผนภูมิและการวิเคราะห์ข้อมูล
- `Pillow>=10.0.0`: ประมวลผลรูปภาพกราฟฟิก

---

## 5. การทดสอบและรันโปรแกรม (Execution & Verification)

### 5.1 ทดสอบรัน Demo แบบรวดเร็ว (Quick Verification Demo)
รันตัวแก้ปัญหา MILP กับชุดคำขอจองคิวจริง (Real Pending Requests N=10):

```bash
python van_allocation_engine.py --mode demo
```

**ผลลัพธ์ที่คาดหวัง:**
- ตัวแก้ปัญหา CBC จะรายงานสถานะ `Optimal`
- แสดงรายงาน JSON สรุปการจัดสรรรถ, การยืมรถข้ามคณะ และการตรวจสอบเกณฑ์ HOS โดยไม่มีข้อผิดพลาด (Safety violations = 0)
- เวลาประมวลผลประมาณ 0.2 - 0.5 วินาที

```json
{
  "method": "MILP",
  "solver_status": "Optimal",
  "total_requests": 10,
  "fully_assigned": 9,
  "fulfillment_rate_pct": 90.0,
  "safety_violations": 0
}
```

---

### 5.2 ทดสอบรันการจำลองและบททดสอบทั้งหมด (Full Experiment Suite)
รันสคริปต์ทดสอบมาตรฐาน 69 Test Cases และการทดลองความอ่อนไหวทั้งหมด:

```bash
python testing_v8_limtw60.py
```

เมื่อรันเสร็จสิ้น โปรแกรมจะสร้างไฟล์รายงานผลลัพธ์ฉบับสมบูรณ์ที่:
`report_v8.md`

---

## 6. ข้อกำหนดรูปแบบข้อมูลนำเข้า (Input Data Specification)

หากต้องการนำชุดข้อมูลจริงของหน่วยงานมาประมวลผล ให้จัดเตรียมไฟล์ CSV ในโฟลเดอร์ `raw_data/` ตามโครงสร้างดังนี้:

### 6.1 ตารางประวัติการจองรถ (`*_booking*.csv`)
| คอลัมน์ | ชนิดข้อมูล | คำอธิบาย | ตัวอย่างค่า |
| :--- | :--- | :--- | :--- |
| `booking_id` | Text | รหัสคำขอจอง | `ICT_BK000447` |
| `requester_faculty` | Text | คณะผู้ขอใช้บริการ | `ICT`, `SCI`, `SEEN` |
| `passenger_count` | Integer | จำนวนผู้โดยสาร (1-33 คน) | `11` |
| `trip_scope` | Text | ขอบเขตเดินทาง (`in_province` / `out_of_province`) | `out_of_province` |
| `start_time` | Datetime | วันและเวลาเริ่มต้นทริป | `2026-06-27 13:00:00` |
| `end_time` | Datetime | วันและเวลาสิ้นสุดทริป | `2026-06-27 22:00:00` |

### 6.2 ตารางคนขับและรถ (`*_drivers*.csv`)
| คอลัมน์ | ชนิดข้อมูล | คำอธิบาย | ตัวอย่างค่า |
| :--- | :--- | :--- | :--- |
| `driver_id` | Text | รหัสประจำตัวพนักงานขับรถ | `SCI_DRIVER_NOW` |
| `faculty` | Text | คณะต้นสังกัด | `SCI` |
| `start_time` | Datetime | วันเวลาเริ่มงานจริง/งานคงที่ | `2026-06-27 08:00:00` |
| `end_time` | Datetime | วันเวลาสิ้นสุดงานจริง | `2026-06-27 17:00:00` |

---

## 7. กฎความปลอดภัย Hours-of-Service (HOS Safety Rules)

ระบบบังคับใช้เงื่อนไขความปลอดภัยอย่างเคร่งครัด (Hard Constraints) ดังนี้:

1. **Daily Duty Limit ($H_{\max}^{\text{day}} \le 14.0$ ชั่วโมง):**
   - พนักงานขับรถ 1 คน ไม่สามารถปฏิบัติงานรวมเกิน 14 ชั่วโมงในรอบ 24 ชั่วโมง
2. **Weekly Cumulative Limit ($H_{\max}^{\text{week}} \le 60.0$ ชั่วโมง):**
   - ชั่วโมงการทำงานสะสมย้อนหลัง 7 วัน (Rolling 7-day Window) ต้องไม่เกิน 60.0 ชั่วโมง
   - ป้องกันการเกิดความล้าสะสมจากการขับรถติดต่อกันหลายวัน
3. **Turnaround Rest Limit ($\Delta_{\text{rest}} \ge 10.0$ ชั่วโมง):**
   - ต้องมีช่วงเวลาพักผ่อนติดต่อกันไม่น้อยกว่า 10.0 ชั่วโมง ระหว่างการสิ้นสุดงานก่อนหน้ากับการเริ่มงานถัดไป

---

## 8. การแก้ปัญหาที่พบบ่อย (Troubleshooting & FAQ)

### Q1: พบข้อผิดพลาด `FileNotFoundError: ไม่พบไฟล์ข้อมูลเลยใน raw_data`
- **สาเหตุ:** สคริปต์ไม่พบโฟลเดอร์ `raw_data` ในไดเรกทอรีปัจจุบัน
- **วิธีแก้:** ตรวจสอบว่าได้รันคำสั่งจากภายในโฟลเดอร์ `soure_code/` หรือตรวจสอบว่ามีโฟลเดอร์ `soure_code/raw_data/` ครบถ้วน

### Q2: พบข้อผิดพลาด `PulpSolverError: Cannot find cbc.exe`
- **สาเหตุ:** ไลบรารี `pulp` ติดตั้งไม่สมบูรณ์
- **วิธีแก้:** รันคำสั่ง `pip install --force-reinstall pulp` อีกครั้ง เพื่อให้ดาวน์โหลด CBC binary ของระบบปฏิบัติการที่ใช้อัตโนมัติ

### Q3: ปัญหาภาษาไทยหรือปี พ.ศ. ในไฟล์ CSV
- **คำอธิบาย:** ระบบมีฟังก์ชัน Preprocessing ทำความสะอาดอัตโนมัติ (`preclean_be_date_str`) ซึ่งจะแปลงปี พ.ศ. (เช่น 2567) เป็น ค.ศ. (2024) อัตโนมัติก่อนส่งเข้าโมเดล

---
**จัดทำและดูแลโดย:** คณะผู้วิจัยโครงการระบบจัดสรรรถตู้โดยสารข้ามคณะ (Van Allocation Model)
