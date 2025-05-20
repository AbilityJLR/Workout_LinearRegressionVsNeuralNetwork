# การพยากรณ์แคลอรีที่เผาผลาญของสมาชิกฟิตเนส

ที่เก็บโค้ดนี้เป็นโปรเจกต์ Machine Learning สำหรับทำนายจำนวนแคลอรีที่เผาผลาญระหว่างการออกกำลังกาย โดยใช้โครงข่ายประสาทเทียมแบบ Multi-Layer Perceptron (MLP) พัฒนาด้วย Keras และ TensorFlow

## สารบัญ

* [ภาพรวมโปรเจกต์](#ภาพรวมโปรเจกต์)
* [ชุดข้อมูล](#ชุดข้อมูล)
* [วิธีการดำเนินงาน](#วิธีการดำเนินงาน)
* [สถาปัตยกรรมของโมเดล](#สถาปัตยกรรมของโมเดล)
* [การติดตั้ง](#การติดตั้ง)
* [การใช้งาน](#การใช้งาน)
* [ผลลัพธ์](#ผลลัพธ์)
* [โครงสร้างไฟล์](#โครงสร้างไฟล์)
* [ไลบรารีที่ใช้](#ไลบรารีที่ใช้)
* [งานในอนาคต](#งานในอนาคต)
* [สิทธิ์การใช้งาน](#สิทธิ์การใช้งาน)

## ภาพรวมโปรเจกต์

การติดตามการออกกำลังกายและแคลอรีที่เผาผลาญเป็นสิ่งสำคัญสำหรับการวางแผนฟิตเนสและการจัดการน้ำหนัก โครงการนี้พัฒนาโครงข่ายประสาทเทียม MLP เพื่อทำนายจำนวนแคลอรีที่เผาผลาญจากคุณลักษณะต่างๆ เช่น ระยะเวลา อัตราการเต้นของหัวใจ อุณหภูมิร่างกาย อายุ เพศ และอื่นๆ

## ชุดข้อมูล

ใช้ชุดข้อมูล **Gym Members Exercise Tracking** จาก Kaggle ซึ่งมีข้อมูลทั้งหมด 973 ระเบียน โดยมีฟีเจอร์ดังนี้:

* `User_ID`: รหัสสมาชิก (จะถูกตัดออกก่อนนำเข้าสู่โมเดล)
* `Gender`: เพศ (ชาย/หญิง) แปลงเป็น one-hot encoding
* `Age`, `Height`, `Weight`
* `Duration` (นาที)
* `Heart_Rate` (จำนวนครั้งต่อนาทีเฉลี่ย)
* `Body_Temp` (°C)
* `Calories_Burned`: ตัวแปรเป้าหมาย

## วิธีการดำเนินงาน

1. **การเตรียมข้อมูล**

   * ลบคอลัมน์รหัสสมาชิก
   * แปลง `Gender` เป็น one-hot encoding
   * แบ่งข้อมูลเป็นชุดฝึก (80%) และชุดทดสอบ (20%)
   * ปรับสเกลฟีเจอร์ด้วย `StandardScaler`
2. **โมเดลฐาน (Baseline)**

   * Linear Regression สำหรับเปรียบเทียบเบื้องต้น
3. **โครงข่ายประสาทเทียม**

   * สร้างและคอมไพล์ MLP ด้วย Keras
   * ฝึกอบรม 100 รอบ (epochs) พร้อมการแบ่ง validation
4. **การประเมินผล**

   * Mean Squared Error (MSE)
   * R-squared (R²)
   * การวาดกราฟ Learning Curves และ Actual vs. Predicted

## สถาปัตยกรรมของโมเดล

```
Input Layer: 13 ฟีเจอร์
Hidden Layer 1: 256 ยูนิต, ReLU
Hidden Layer 2: 128 ยูนิต, ReLU
Hidden Layer 3: 64 ยูนิต, ReLU
Output Layer: 1 ยูนิต, ReLU
Optimizer: Adam (lr=0.001)
Loss: Mean Squared Error
```

## การติดตั้ง

1. โคลนที่เก็บโค้ด:

   ```bash
   git clone https://github.com/<your-username>/gym-calories-prediction.git
   cd gym-calories-prediction
   ```
2. สร้าง virtual environment และติดตั้งไลบรารี:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

## การใช้งาน

เปิด Jupyter Notebook และรันไฟล์โปรเจกต์:

```bash
jupyter notebook [CN340]_FinalProject.ipynb
```

จากนั้นทำตามขั้นตอนในโน้ตบุ๊กเพื่อเตรียมข้อมูล ฝึกโมเดล และดูผลลัพธ์กราฟต่างๆ

## ผลลัพธ์

* **Neural Network**: MSE = 1001.19, R² = 0.985
* **Linear Regression**: MSE = 3021.45, R² = 0.912

กราฟ Learning Curves แสดงการเรียนรู้ที่เสถียร ไม่มีการ overfitting และกราฟ Scatter Plot แสดงค่าทำนายเทียบกับค่าจริงที่สอดคล้องกันอย่างใกล้เคียง

## โครงสร้างไฟล์

```
├── Report_CN340.pdf             # รายงานโปรเจกต์
├── [CN340]_FinalProject.ipynb   # Jupyter Notebook
├── data/                        # ไฟล์ชุดข้อมูล (CSV)
├── models/                      # โมเดลและ scaler ที่บันทึกไว้
├── requirements.txt             # ไลบรารีที่ต้องติดตั้ง
└── README.md                    # ไฟล์นี้
```

## ไลบรารีที่ใช้

* Python 3.8+
* pandas
* numpy
* scikit-learn
* tensorflow (รวม Keras)
* matplotlib

## งานในอนาคต

* ปรับแต่ง hyperparameters ด้วย Grid/Randomized Search
* ทดลองสถาปัตยกรรมลึกขึ้นหรือลักษณะอื่นๆ
* เพิ่มฟีเจอร์ใหม่ (เช่น ประเภทกิจกรรม ดัชนีมวลกาย)
* ทำเป็นแอปเว็บหรือแอปมือถือ

---

*ผู้พัฒนา:*

```
  6510742098 tanakrit maenphol
  6510742254 sorayuth ingboon
  6510742510 montira innoy
```

สำหรับโปรเจกต์ CN340
