# คู่มือการใช้งานซอร์สโค้ดและระบบจัดสรรรถตู้โดยสารส่วนกลางข้ามคณะด้วยโมเดล MILP (ฉบับภาษาไทย)

**ชื่อระบบ:** Cross-Faculty Shared Passenger Van Allocation MILP Engine (รุ่น v6 Empirical Pipeline)  
**ชุดซอร์สโค้ดหลัก:** `van_allocation_engine.py` และ `testing_v6_Candidate.ipynb`  
**มาตรฐานความปลอดภัยที่รองรับ:** พระราชบัญญัติการขนส่งทางบก พ.ศ. 2522, พระราชบัญญัติจราจรทางบก พ.ศ. 2522 และเกณฑ์ชั่วโมงการทำงานและการพักผ่อน (Hours-of-Service: HOS 14/72/10 ชั่วโมง)

---

## 1. ภาพรวมของระบบ (System Overview)

ซอร์สโค้ดชุดนี้เป็นระบบสนับสนุนการตัดสินใจ (Decision Support System: DSS) และเครื่องมือคำนวณเชิงคณิตศาสตร์ด้วยโปรแกรมเชิงเส้นแบบผสมจำนวนเต็ม (Mixed-Integer Linear Programming: MILP) สำหรับแก้ปัญหาการจัดสรรรถตู้โดยสารและพนักงานขับรถร่วมกันระหว่างหลายคณะภายในมหาวิทยาลัยในกรอบเวลาวางแผนล่วงหน้า 14 วัน (Rolling 14-Day Horizon)

### ขีดความสามารถหลักของซอร์สโค้ด
1. **การจัดสรรทรัพยากรข้ามคณะตามลำดับขั้น (Hierarchical Cross-Faculty Pooling):** ให้ความสำคัญกับการใช้รถและคนขับของคณะเจ้าของคำขอก่อน และอนุญาตให้ยืมรถข้ามคณะอัตโนมัติเมื่อรถของคณะตนเองไม่ว่างหรือไม่เพียงพอ
2. **การแบ่งกลุ่มผู้โดยสารหลายคัน (Multi-Van Request Splitting):** รองรับการกระจายผู้โดยสารจากคำขอขนาดใหญ่ (เช่น 12–33 คน) ลงในรถตู้หลายคัน (ความจุมาตรฐาน 11 ที่นั่ง/คัน สูงสุดไม่เกิน $K_r = 3$ คัน)
3. **การบังคับใช้กฎความปลอดภัยด้านความเหนื่อยล้าของคนขับ (Hard HOS Constraints):**
   - ชั่วโมงปฏิบัติงานรวมสูงสุดไม่เกิน **14 ชั่วโมง/วัน** ($H_{\max}^{\text{day}} = 14$)
   - ชั่วโมงปฏิบัติงานสะสมรวมไม่เกิน **72 ชั่วโมง/14 วัน** ($H_{\max}^{14\text{d}} = 72$ หรือเฉลี่ย 36 ชั่วโมง/สัปดาห์)
   - ระยะเวลาพักผ่อนขั้นต่ำระหว่างทริปไม่น้อยกว่า **10 ชั่วโมงต่อเนื่อง** ($\Delta_{\text{rest}} = 10$)
4. **ระบบทำความสะอาดข้อมูลจริงอัตโนมัติ (Automated Data Preprocessing Pipeline):**
   - แปลงวันที่ปีพุทธศักราช (`พ.ศ. > 2400`) เป็นคริสต์ศักราช (`ค.ศ.`) อัตโนมัติก่อนแปลงเป็น Timestamp เพื่อป้องกันข้อผิดพลาด `OutOfBoundsDatetime`
   - แปลงรูปแบบเวลาที่ใช้จุดทศนิยม (เช่น `10.00`, `16.30`) ให้เป็นมาตรฐาน ISO (`10:00:00`, `16:30:00`)
   - ตัดแถวว่างท้ายตาราง Excel (`.dropna(how='all')`) และปรับมาตรฐานคำศัพท์สถานะคำขอ (`approve` $\rightarrow$ `approved`, `delete` $\rightarrow$ `cancel`)
5. **ชุดอัลกอริทึมเปรียบเทียบ (Baseline Benchmarks) และตัวตรวจสอบความถูกต้อง (Verification Oracle):**
   - `MILP` (โมเดลที่นำเสนอ แก้ด้วยตัวแก้ปัญหา COIN-OR CBC ผ่านไลบรารี PuLP)
   - `FCFS-A` (จัดสรรตามลำดับการจองแบบไม่ตรวจโควตาความเหนื่อยล้าสะสม)
   - `FCFS-B` (จัดสรรตามลำดับการจองภายใต้กรอบความปลอดภัย HOS ครบถ้วน)
   - `Heuristic-SDF` (จัดสรรโดยเรียงลำดับทริปสั้นและจำนวนผู้โดยสารน้อยก่อนภายใต้กฎ HOS)
   - `Rule-Based Verification Oracle` (ตัวตรวจสอบความถูกต้องเชิงตรรกะอิสระ 864 สถานการณ์)

---

## 2. โครงสร้างไฟล์ในโปรเจกต์ (Project File Structure)

| ชื่อไฟล์ / โฟลเดอร์ | ประเภท | คำอธิบายหน้าที่ |
| :--- | :--- | :--- |
| `van_allocation_engine.py` | Python Module / CLI | ซอร์สโค้ดหลักแบบโมดูลาร์สำหรับรันผ่าน Command Line หรือนำเข้า (`import`) ในระบบเว็บ/ซอฟต์แวร์ของมหาวิทยาลัย |
| `testing_v6_Candidate.ipynb` | Jupyter Notebook | สมุดบันทึกการทดลองฉบับสมบูรณ์ (70 เซลล์) สำหรับทวนสอบผลลัพธ์ทั้งหมดในบทความวิจัย |
| `requirements.txt` | Configuration | รายการไลบรารี Python ที่จำเป็นสำหรับการติดตั้ง |
| `bookingN.csv`, `driversN.csv` | Input CSV | ข้อมูลคำขอใช้รถและตารางงานคนขับจริงของคณะ ICT |
| `sci_bookingN.csv`, `sci_driversN.csv` | Input CSV | ข้อมูลคำขอใช้รถและตารางงานคนขับจริงของคณะวิทยาศาสตร์ (SCI) |
| `seen_bookingN.csv`, `seen_driversN.csv` | Input CSV | ข้อมูลจำลองสภาวะวิกฤต (Stress-test) ของคณะพลังงานและสิ่งแวดล้อม (SEEN) |
| `raw_output/` | Output Directory | โฟลเดอร์เก็บตารางผลลัพธ์ `.csv` และกราฟ `.png` ที่สร้างจากการรันโมเดล |
| `EASR_Submission/` | Submission Package | โฟลเดอร์บรรจุไฟล์ต้นฉบับภาษาอังกฤษตามเทมเพลตวารสาร EASR (`EASR_manuscript_EN.docx`) และภาพความละเอียด 300 DPI (`.jpg`, `.tiff`) |

---

## 3. ความต้องการของระบบและการติดตั้ง (System Requirements & Installation)

### 3.1 ความต้องการขั้นต่ำของระบบ
- **ระบบปฏิบัติการ:** Windows 10/11, macOS หรือ Linux (Ubuntu 20.04+)
- **Python:** เวอร์ชัน 3.9 ถึง 3.12 (แนะนำ Python 3.11)
- **หน่วยความจำ (RAM):** ขั้นต่ำ 4 GB (แนะนำ 8 GB สำหรับการรันจำลองสถานการณ์ 900 รอบ)
- **ตัวแก้ปัญหา (MILP Solver):** ตัวแก้ปัญหา **COIN-OR CBC** ซึ่งติดตั้งมาพร้อมกับแพ็กเกจ `pulp` โดยอัตโนมัติ (ไม่ต้องซื้อไลเซนส์เพิ่มเติม)

### 3.2 ขั้นตอนการติดตั้ง
เปิดโปรแกรม Terminal หรือ PowerShell ในโฟลเดอร์โปรเจกต์ (`d:\Pro`) แล้วรันคำสั่งต่อไปนี้:

```powershell
# 1. สร้างสภาพแวดล้อมเสมือน (Virtual Environment - ทางเลือกที่แนะนำ)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. ติดตั้งไลบรารีทั้งหมดตามไฟล์ requirements.txt
pip install -r requirements.txt
```

---

## 4. รูปแบบไฟล์ข้อมูลนำเข้า (Input Data Schema)

เพื่อให้มหาวิทยาลัยหรือหน่วยงานอื่นสามารถนำข้อมูลของตนเองเข้าสู่ระบบได้ทันที ข้อมูลนำเข้าแบ่งออกเป็น 2 ตารางหลักในรูปแบบ CSV (เข้ารหัส `UTF-8`):

### 4.1 ไฟล์คำขอจองรถ (`*bookingN.csv`)
| ชื่อคอลัมน์ | ชนิดข้อมูล | ตัวอย่างค่า | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `booking_id` | String | `B000001` | รหัสคำขอใช้รถ (ระบบจะเติมรหัสคณะนำหน้าอัตโนมัติเพื่อกันรหัสซ้ำ) |
| `requester_faculty` | String | `ICT`, `SCI` | ชื่อย่อของคณะผู้ยื่นคำขอ |
| `passenger_count` | Integer | `8`, `15` | จำนวนผู้โดยสารรวม ($P_r \ge 1$) |
| `trip_scope` | String | `in_province`, `out_of_province` | ขอบเขตการเดินทาง (`out_of_province` บังคับใช้คนขับรถตู้ประจำทุกคัน) |
| `start_date`, `end_date` | Date String | `2025-02-10` หรือ `10/02/2568` | วันที่เริ่มต้นและสิ้นสุดทริป (รองรับทั้ง ค.ศ. และ พ.ศ.) |
| `start_time`, `end_time` | Time String | `08:00:00` หรือ `08.00` | เวลาเริ่มต้นและสิ้นสุดทริป (รองรับทั้ง `:` และ `.`) |
| `booking_status` | String | `approved`, `pending`, `cancel` | สถานะคำขอ (ระบบแปลง `approve` $\rightarrow$ `approved` อัตโนมัติ) |

### 4.2 ไฟล์บันทึกตารางงานคนขับ (`*driversN.csv`)
| ชื่อคอลัมน์ | ชนิดข้อมูล | ตัวอย่างค่า | คำอธิบาย |
| :--- | :--- | :--- | :--- |
| `log_id` | String | `DL000001` | รหัสบันทึกการปฏิบัติงานของพนักงานขับรถ |
| `driver_id` | String | `ICT_DRIVER_NOW` | รหัสพนักงานขับรถ (`_NOW` = คนขับปัจจุบันที่ใช้บล็อกตารางงาน, `_OLD` = ข้อมูลสถิติในอดีต) |
| `work_date` | Date String | `15/01/2568` | วันที่ปฏิบัติงาน (รองรับทั้ง ค.ศ. และ พ.ศ.) |
| `start_time`, `end_time` | Time String | `08.30`, `17.00` | เวลาเริ่มงานและเลิกงานในวันนั้น |
| `work_hours` | Float | `8.5` | จำนวนชั่วโมงทำงานในวันนั้น |

---

## 5. วิธีการรันใช้งานซอร์สโค้ด (Execution Guide)

### วิธีที่ 1: รันผ่านสคริปต์โมดูลหลัก (`van_allocation_engine.py`)
เหมาะสำหรับผู้ดูแลระบบหรือการเชื่อมต่อกับเว็บแอปพลิเคชัน:

```powershell
# 1. แสดงตารางสรุปผลการทดลองเชิงประจักษ์ v6 ทั้งหมดจากโฟลเดอร์ raw_output/
python van_allocation_engine.py --mode summary

# 2. รันการสาธิตการจัดสรรรถตู้ข้ามคณะแบบจำลองตัวเอง (Self-Contained MILP Demo)
python van_allocation_engine.py --mode demo
```

### วิธีที่ 2: การเรียกใช้งานฟังก์ชัน MILP ในโค้ด Python อื่น (API Integration)
สามารถนำเข้าฟังก์ชัน `solve_van_allocation_milp` และ `AllocationConfig` ไปใช้ในระบบจองรถของมหาวิทยาลัยได้โดยตรง:

```python
import pandas as pd
from van_allocation_engine import AllocationConfig, solve_van_allocation_milp

# 1. กำหนดค่าพารามิเตอร์สำหรับมหาวิทยาลัยของคุณ
config = AllocationConfig(
    faculties=["ENG", "MED", "BUS", "SCI"],
    van_capacity=11,
    max_hours_per_day=14.0,
    max_hours_per_14d=72.0,
    min_rest_hours=10.0,
)

# 2. เตรียมข้อมูลรถตู้ คนขับ และคำขอใช้รถ
vans = [
    {"van_id": "VAN_ENG_01", "faculty": "ENG", "capacity": 11, "primary_driver_id": "DRV_ENG_01"},
    {"van_id": "VAN_SCI_01", "faculty": "SCI", "capacity": 11, "primary_driver_id": "DRV_SCI_01"},
]
drivers = [
    {"driver_id": "DRV_ENG_01", "faculty": "ENG", "base_day_hours": {1: 4.0}},
    {"driver_id": "DRV_SCI_01", "faculty": "SCI", "base_day_hours": {1: 0.0}},
]
requests = [
    {
        "request_id": "REQ_101",
        "faculty": "ENG",
        "pax": 10,
        "max_vans": 1,
        "is_out_of_province": 1,
        "start_dt": pd.Timestamp("2025-03-01 08:00:00"),
        "end_dt": pd.Timestamp("2025-03-02 18:00:00"),
        "day_hours": {1: 10.0, 2: 10.0},
    }
]

# 3. สั่งรันตัวแก้ปัญหา MILP
result = solve_van_allocation_milp(requests, vans, drivers, config=config)
print("สถานะคำตอบ:", result["solver_status"])
print("ผลการจัดสรร:", result["assignments"])
```

### วิธีที่ 3: รันการทดลองทั้งหมดซ้ำผ่าน Jupyter Notebook (`testing_v6_Candidate.ipynb`)
หากต้องการรันการทดลองทั้ง 70 เซลล์ใหม่ตั้งแต่ต้นจนจบเพื่อสร้างไฟล์ CSV และกราฟทั้งหมดใน `raw_output/`:

```powershell
# เปิดด้วย Jupyter Notebook / VS Code หรือสั่งรันอัตโนมัติผ่าน nbclient:
jupyter nbconvert --to notebook --execute --inplace testing_v6_Candidate.ipynb
```

---

## 6. การปรับตั้งค่าพารามิเตอร์สำหรับมหาวิทยาลัยอื่น (Customization Guide)

มหาวิทยาลัยอื่นสามารถปรับแก้ค่าพารามิเตอร์ในคลาส `AllocationConfig` ในไฟล์ `van_allocation_engine.py` ได้ทันทีโดยไม่ต้องแก้ไขสมการคณิตศาสตร์:

1. **ความจุรถตู้ (`van_capacity`):** ค่าเริ่มต้นคือ `11` ที่นั่ง หากมหาวิทยาลัยใช้รถตู้รุ่น 13 ที่นั่ง หรือรถมินิบัส 20 ที่นั่ง สามารถเปลี่ยนค่า `van_capacity` หรือระบุ `capacity` รายคันใน `vans` ได้ทันที
2. **เพดานชั่วโมงทำงานและความปลอดภัย (`max_hours_per_day`, `max_hours_per_14d`, `min_rest_hours`):**
   - ค่าเริ่มต้นตั้งไว้ที่ `14.0` ชม./วัน, `72.0` ชม./14 วัน และพักอย่างน้อย `10.0` ชม.
   - หากมหาวิทยาลัยต้องการใช้เกณฑ์ที่เข้มงวดขึ้น (เช่น ไม่เกิน `10.0` ชม./วัน หรือ `60.0` ชม./14 วัน) สามารถปรับตัวเลขได้โดยตรง
3. **น้ำหนักลำดับความสำคัญในสมการเป้าหมาย (`w3_same_faculty_van` vs. `w3_cross_faculty_van`):**
   - ค่าเริ่มต้นกำหนดให้ใช้รถคณะตนเองมีต้นทุนแฝง `100.0` และยืมรถข้ามคณะมีต้นทุนแฝง `400.0` เพื่อบังคับให้ใช้รถคณะตนเองก่อนเสมอ
   - หากมหาวิทยาลัยเปลี่ยนเป็นระบบ **ส่วนกลางสมบูรณ์ (Centralized Pool)** ที่ไม่มีการแบ่งแยกคณะ ให้ตั้งค่า `w3_same_faculty_van = 100.0` และ `w3_cross_faculty_van = 100.0` เท่ากัน

---

## 7. คำอธิบายไฟล์ผลลัพธ์ในโฟลเดอร์ `raw_output/`

| ชื่อไฟล์ผลลัพธ์ | คำอธิบายเนื้อหา |
| :--- | :--- |
| `empirical_probability_comparison.csv` | เปรียบเทียบสัดส่วนความน่าจะเป็นเชิงประจักษ์จากข้อมูลจริง (ICT+SCI) กับค่าสมมติฐานเริ่มต้น |
| `scenario_empirical_vs_synthetic.csv` | สรุปผลการจัดสรรครบ 864 สถานการณ์ (97.09% จัดสรรสำเร็จบริบูรณ์ภายใต้การถ่วงน้ำหนักข้อมูลจริง) |
| `baseline_comparison_real.csv` | ผลการเปรียบเทียบ `MILP`, `FCFS-A`, `FCFS-B` และ `Heuristic-SDF` บนคำขอรออนุมัติจริง ($N=10$) |
| `baseline_comparison_synthetic.csv` | ผลการเปรียบเทียบอัลกอริทึมทั้ง 4 ตัวบนชุดข้อมูลจำลอง 900 สถานการณ์ |
| `exp1_scalability.csv` | เวลาประมวลผลและอัตราการได้คำตอบที่เหมาะสมที่สุดเมื่อเพิ่มจำนวนคำขอตั้งแต่ $N=5$ ถึง $60$ |
| `exp2_fleet_sizing_and_sharing.csv` | ผลการวิเคราะห์ความไวต่อจำนวนรถตู้ต่อคณะ (1, 2, 3 คัน/คณะ) และการเปิด/ปิดการแชร์รถข้ามคณะ |
| `exp3_hos_sensitivity.csv` | ผลการวิเคราะห์ความไวเมื่อปรับเพดานชั่วโมงทำงานสะสม 14 วันตั้งแต่ 48 ถึง 96 ชั่วโมง |
| `baseline_fulfillment.png` | กราฟเปรียบเทียบอัตราการจัดสรรสำเร็จและจำนวนการละเมิดความปลอดภัยของทั้ง 4 อัลกอริทึม |
| `sensitivity_and_scalability.png` | กราฟ 4 พาเนลแสดงผล Fleet Sizing, HOS Sensitivity และเวลาประมวลผล (Scalability) |

---

## 8. การแก้ไขปัญหาเบื้องต้น (Troubleshooting)

1. **เกิดข้อผิดพลาด `OutOfBoundsDatetime` เมื่อนำเข้าไฟล์ CSV จาก Excel ภาษาไทย:**
   - **สาเหตุ:** พ.ศ. 2566–2568 เกินขีดจำกัดปี ค.ศ. 2262 ของชนิดข้อมูล `datetime64[ns]` ในไลบรารี Pandas
   - **วิธีแก้ไข:** เรียกใช้ฟังก์ชัน `parse_flexible_date(df['date_col'])` ใน `van_allocation_engine.py` ซึ่งจะแปลงปี พ.ศ. (`> 2400`) ให้เป็น ค.ศ. ก่อนเข้าสู่ `pd.to_datetime` เสมอ
2. **จำนวนแถวของข้อมูลคำขอมีมากผิดปกติ (เช่น 997 แถวแทนที่จะเป็น 156 แถว):**
   - **สาเหตุ:** ไฟล์ CSV ที่บันทึกจาก Excel มีแถวว่างที่มีเฉพาะเครื่องหมายคอมม่า (`,,,,,`) ค้างอยู่ท้ายไฟล์
   - **วิธีแก้ไข:** ใช้คำสั่ง `df = df.dropna(how='all')` ก่อนประมวลผล (ดำเนินการให้อัตโนมัติแล้วใน v6)
3. **ไม่พบตัวแก้ปัญหา CBC (`PulpSolverError: Cannot execute cbc`):**
   - **วิธีแก้ไข:** รันคำสั่ง `pip install --upgrade --force-reinstall pulp` เพื่อให้แพ็กเกจติดตั้งไฟล์ไบนารีของ CBC สำหรับระบบปฏิบัติการของคุณใหม่อีกครั้ง
