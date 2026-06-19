import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. ตั้งค่าหน้าตาเว็บให้กว้างและทันสมัย ---
st.set_page_config(
    page_title="Pineapple Brix & Angle Analyzer",
    page_icon="🍍",
    layout="wide"
)

# แทรก CSS เล็กน้อยเพื่อตกแต่งให้สวยงามขึ้น
st.markdown("""
    <style>
    .main-title { text-align: center; color: #E67E22; font-weight: bold; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #7F8C8D; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🍍 ระบบวิเคราะห์ตาสับปะรด & ประเมินค่า Brix</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>อัปโหลดภาพถ่ายสับปะรดเพื่อคำนวณมุมหลัก (Main Spiral) และประมาณการค่าความหวานอัตโนมัติ</p>", unsafe_allow_html=True)
st.write("---")

# --- 2. ฟังก์ชันหลักสำหรับประมวลผลภาพ (ยกมาจากโค้ด Colab ของคุณ) ---
def analyze_pineapple(pil_image, model_version, force_left_to_right):
    # แปลงจาก PIL Image (Streamlit) เป็น OpenCV BGR
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1) Resize ภาพถ้าใหญ่เกินไป
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    
    # สร้างภาพสำหรับวาดแสดงผล (แปลงเป็น RGB รอไว้เลย)
    display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2) Preprocess
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    
    # 3) Find Contours (หาตาสับปะรด)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / (ch + 1e-6)
        if ratio < 0.4 or ratio > 2.5:
            continue
        centers.append([x + cw/2, y + ch/2])
        
    # ตรวจสอบจำนวนจุด
    if len(centers) < 10:
        return None, "พบตาสับปะรดน้อยเกินไป (น้อยกว่า 10 จุด) กรุณาเปลี่ยนรูปภาพหรือเช็กแสงขอบภาพ"
        
    pts = np.array(centers, dtype=np.float32)
    pts = pts[np.argsort(pts[:,0] + pts[:,1])] # เรียงจุด
    pts = pts[:50] # ใช้ 50 จุดแรก
    
    if not force_left_to_right:
        pts[:,0] = -pts[:,0]
        
    # 4) วาดเส้นเชื่อมโยงระหว่างจุด (Connections)
    for i in range(len(pts)-1):
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
        cv2.line(display_rgb, p1, p2, (255, 255, 0), 2)
        
    # 5) Fit Main Spiral Line
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = map(float, [vx, vy, x0, y0])
    
    angle_main = np.degrees(np.arctan2(vy, vx))
    if angle_main < 0:
        angle_main += 180
        
    L = 3000
    brix = None
    angle_used = None
    
    # 6) คำนวณตามโมเดลที่เลือก
    if model_version == "model2":
        # วาดเส้นหลัก (น้ำเงิน)
        cv2.line(display_rgb, (int(x0-vx*L), int(y0-vy*L)), (int(x0+vx*L), int(y0+vy*L)), (0, 0, 255), 6)
        
        theta = np.radians(75)
        vx_r = vx*np.cos(theta) - vy*np.sin(theta)
        vy_r = vx*np.sin(theta) + vy*np.cos(theta)
        
        # วาดเส้นหมุน (แดง)
        cv2.line(display_rgb, (int(x0-vx_r*L), int(y0-vy_r*L)), (int(x0+vx_r*L), int(y0+vy_r*L)), (255, 0, 0), 5)
        
        angle_rot = np.degrees(np.arctan2(vy_r, vx_r))
        if angle_rot < 0:
            angle_rot += 180
        angle_used = angle_rot
        
        # คำนวณ Brix Model 2
        x_brix = abs(angle_main - 146)
        brix = 0.0428*(x_brix**2) - 0.9296*x_brix + 16.037
        
    elif model_version == "model3":
        # วาดเส้นหลัก (น้ำเงิน)
        cv2.line(display_rgb, (int(x0-vx*L), int(y0-vy*L)), (int(x0+vx*L), int(y0+vy*L)), (0, 0, 255), 6)
        
        theta = np.radians(98)
        vx_r = vx*np.cos(theta) - vy*np.sin(theta)
        vy_r = vx*np.sin(theta) + vy*np.cos(theta)
        
        # วาดเส้นหมุน (แดง)
        cv2.line(display_rgb, (int(x0-vx_r*L), int(y0-vy_r*L)), (int(x0+vx_r*L), int(y0+vy_r*L)), (255, 0, 0), 6)
        
        angle_98 = np.degrees(np.arctan2(vy_r, vx_r))
        if angle_98 < 0:
            angle_98 += 180
        angle_used = angle_98
        
        # วาดเส้นแนวระดับ (เหลือง)
        cv2.line(display_rgb, (int(x0-L), int(y0)), (int(x0+L), int(y0)), (255, 255, 0), 3)
        
        # คำนวณ Brix Model 3
        x_brix = abs(angle_main - 135)
        brix = 0.0366*(x_brix**2) - 0.8924*x_brix + 16.696

    # 7) วาดจุดตาสับปะรด (วงกลมเขียว) ลงบนภาพผลลัพธ์
    for p in pts:
        cv2.circle(display_rgb, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
        
    # วาดข้อความผลลัพธ์ลงบนภาพเหมือนต้นฉบับ
    cv2.putText(
        display_rgb, f"Main={angle_main:.1f}  Brix={brix:.2f}%", 
        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3
    )
    
    return {
        "image": display_rgb,
        "angle_main": angle_main,
        "angle_used": angle_used,
        "brix": brix
    }, None


# --- 3. ส่วนควบคุมบนหน้าเว็บ (UI Sidebar & Layout) ---

# ใช้ Sidebar ซ้ายมือสำหรับปรับค่าตัวแปรแทนการพิมพ์ input() ให้ยุ่งยาก
with st.sidebar:
    st.header("⚙️ ตั้งค่าระบบ")
    model_option = st.radio("เลือกโมเดลคำนวณ (Model)", ["model2", "model3"])
    force_lr = st.checkbox("Force Left to Right", value=True)
    st.write("---")
    st.info("💡 คำแนะนำ: เลือกไฟล์ภาพถ่ายของสับปะรดที่เห็นแนวตาชัดเจน เพื่อให้ระบบจับพิกัดได้แม่นยำที่สุด")

# แบ่งหน้าจอเป็น 2 ฝั่ง (ฝั่งซ้าย: อัปโหลดรูปภาพ / ฝั่งขวา: แสดงผลลัพธ์)
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📸 อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพถ่ายสับปะรด...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # เปิดอ่านรูปภาพด้วย PIL
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="รูปภาพต้นฉบับที่อัปโหลด", use_container_width=True)

with col_result:
    st.subheader("📊 ผลการประมวลผล")
    
    if uploaded_file is not None:
        with st.spinner("🔄 ระบบกำลังประมวลผลและวิเคราะห์ตาสับปะรด..."):
            # เรียกใช้ฟังก์ชันประมวลผล
            result, error_msg = analyze_pineapple(input_image, model_option, force_lr)
            
        if error_msg:
            st.error(f"❌ เกิดข้อผิดพลาด: {error_msg}")
        else:
            st.success("✅ ประมวลผลและคำนวณค่าสำเร็จ!")
            
            # แสดงค่าสถิติผ่าน st.metric สวยๆ เหมือนแดชบอร์ด
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(label="📐 Main Spiral Angle", value=f"{result['angle_main']:.2f}°")
            with metric_col2:
                st.metric(label="🔄 Angle Used", value=f"{result['angle_used']:.2f}°")
            with metric_col3:
                st.metric(label="🧪 Predicted Brix", value=f"{result['brix']:.2f} %")
            
            # แสดงรูปภาพที่ผ่านการลากเส้นและจุดวิเคราะห์แล้ว
            st.write("---")
            st.image(result['image'], caption=f"ภาพผลลัพธ์การวิเคราะห์ด้วย {model_option}", use_container_width=True)
            
    else:
        st.info("📌 กรุณาอัปโหลดรูปภาพสับปะรดที่ฝั่งซ้าย ระบบจะคำนวณและแสดงผลลัพธ์ให้ทันที")
