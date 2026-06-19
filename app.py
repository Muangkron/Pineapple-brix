import streamlit as st
import cv2
import numpy as np
import math

# ==========================================
# 1. ฟังก์ชันหลักในการประมวลผลและวิเคราะห์ภาพ
# ==========================================
def analyze_pineapple_pipeline(img):
    """
    ฟังก์ชันรับภาพ BGR จาก OpenCV เข้ามาประมวลผลหาแนวตาสับปะรด
    และคำนวณค่ามุมพร้อมความหวานออกมาโดยอัตโนมัติ
    """
    # ปรับขนาดภาพให้เป็นมาตรฐานเพื่อความแม่นยำในการคำนวณ
    img_resized = cv2.resize(img, (600, 800))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ลดสัญญาณรบกวนของภาพด้วย Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ตรวจจับเส้นขอบตาสับปะรดด้วย Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # ดึงพิกัดของเส้นขอบทั้งหมดออกมาเพื่อนำไปลากเส้นแนวโน้ม (Fit Line)
    points = np.argwhere(edges > 0)
    
    # 🛡️ ระบบเซฟตี้ดักจับ Error: ถ้าจุดพิกเซลมีน้อยเกินไป (ภาพมืด/ไม่ชัด) ให้ดีดออกทันที
    if len(points) < 100:
        return img_resized, 0.0, 0.0, "ระบบไม่สามารถตรวจจับแนวร่องตาสับปะรดได้ชัดเจน กรุณาเปลี่ยนใช้ภาพที่สว่างและเห็นตาเฉียงชัดเจนขึ้นครับ"
    
    # แปลงพิกัดให้อยู่ในรูปแบบ (x, y) สำหรับ OpenCV
    points_xy = np.fliplr(points).astype(np.int32)
    
    # ใช้คำสั่ง fitLine เพื่อหาเวกเตอร์แนวโน้มของตาสับปะรด (vx, vy) และจุดกึ่งกลาง (x0, y0)
    vx, vy, x0, y0 = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # ป้องกันแปลงค่าว่าง: แปลงผลลัพธ์ให้เป็นตัวเลขทศนิยมแบบลอยตัว (Float) อย่างปลอดภัย
    vx = float(vx[0])
    vy = float(vy[0])
    x0 = float(x0[0])
    y0 = float(y0[0])
    
    # คำนวณหามุมของแนวตาสับปะรด (แปลงจากเรเดียนเป็นองศา)
    angle = math.degrees(math.atan2(vy, vx))
    if angle < 0:
        angle += 180  # ปรับมุมให้อยู่ในช่วง 0 - 180 องศาเพื่อความง่าย
        
    # สูตรคำนวณค่าความหวาน Brix (อ้างอิงความสัมพันธ์ตามโครงงานคณิตศาสตร์/ฟิสิกส์)
    # สมมติฐาน: แนวตาสับปะรดที่ทำมุมเฉียงสมบูรณ์ (เช่น ช่วง 45 องศา) จะมีความสมบูรณ์และความหวานสูง
    base_brix = 11.5
    angle_deviation = abs(45 - angle)
    calculated_brix = base_brix + (angle_deviation * 0.08)
    
    # จำกัดช่วงค่าความหวานให้อยู่ในเกณฑ์จริงของสับปะรด (9.0 - 16.0 Brix)
    final_brix = round(min(max(calculated_brix, 9.0), 16.0), 2)
    final_angle = round(angle, 2)
    
    # สร้างภาพผลลัพธ์เพื่อนำไปวาดเส้นแสดงแนวตาสับปะรด
    result_img = img_resized.copy()
    
    # คำนวณจุดเริ่มและจุดสิ้นสุดของเส้นตรงยาวๆ เพื่อวาดทับแนวตาสับปะรด
    line_length = 1000
    pt1 = (int(x0 - vx * line_length), int(y0 - vy * line_length))
    pt2 = (int(x0 + vx * line_length), int(y0 + vy * line_length))
    
    # วาดเส้นแกนกลางตาสับปะรด (เส้นสีแดง ความหนา 3 พิกเซล)
    cv2.line(result_img, pt1, pt2, (0, 0, 255), 3)
    # วาดจุดศูนย์กลางพิกัด (จุดสีเขียว)
    cv2.circle(result_img, (int(x0), int(y0)), 6, (0, 255, 0), -1)
    
    return result_img, final_angle, final_brix, None

# ==========================================
# 2. ส่วนการจัดการหน้าเว็บและ UI ด้วย Streamlit
# ==========================================
st.set_page_config(page_title="Pineapple Brix Detector", page_icon="🍍", layout="wide")

st.title("ระบบวิเคราะห์ความหวานจากตาสับปะรดอัตโนมัติ 🍍")
st.markdown("---")
st.write("💡 **คำแนะนำ:** เพื่อผลลัพธ์ที่แม่นยำ กรุณาใช้ภาพถ่ายสับปะรดในแนวตั้ง มีแสงสว่างเพียงพอ และเห็นลวดลายร่องเฉียงของตาสับปะรดชัดเจน")

# กล่องรับไฟล์ภาพจากผู้ใช้งาน
uploaded_file = st.file_uploader("เลือกรูปภาพสับปะรดของคุณ (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🛠️ แปลงไฟล์ที่อัปโหลดให้กลายเป็น OpenCV BGR ทันที (แก้ปัญหาระบบสีสลับ RGB/BGR ระหว่างเว็บกับ Colab)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # สร้างคอลัมน์ซ้าย-ขวาเพื่อแสดงผลเปรียบเทียบ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 รูปภาพต้นฉบับ")
        # แปลงเป็น RGB เฉพาะตอนแสดงผลบนหน้าเว็บเพื่อให้สีผิวสับปะรดไม่เพี้ยนเป็นสีน้ำเงิน
        st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("📊 ผลการประมวลผลและวิเคราะห์")
        
        # ส่งภาพเข้าสู่กระบวนการคำนวณ
        with st.spinner("กำลังวิเคราะห์แนวตาสับปะรด..."):
            processed_img, angle_result, brix_result, error_msg = analyze_pineapple_pipeline(opencv_img)
            
        if error_msg:
            # แทนที่หน้าจอจะระเบิดเป็นตัวหนังสือสีแดง จะแสดงกล่องคำเตือนสีส้มที่อ่านง่ายแทน
            st.warning(error_msg)
        else:
            # แสดงภาพที่วาดเส้นแนวแกนตาสับปะรดเรียบร้อยแล้ว
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success("วิเคราะห์ข้อมูลสำเร็จ!")
            
            # แสดงแดชบอร์ดค่าตัวเลขที่คำนวณได้แบบสวยงาม
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric(label="📐 มุมแนวตาสับปะรด", value=f"{angle_result} องศา")
            with metrics_col2:
                st.metric(label="🧪 ค่าความหวานโดยประมาณ", value=f"{brix_result} Brix")
