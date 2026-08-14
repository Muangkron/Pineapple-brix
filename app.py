import math
import os
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------------------------
# 1. Config & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pineapple Manual Measurement & Brix Analyzer", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียวสับปะรดแบบกำหนด 2 จุดด้วยตนเอง (Manual 2-Point)")

# -----------------------------------------------------------------------------
# 2. Session State Management
# -----------------------------------------------------------------------------
if "clicked_pts" not in st.session_state:
    st.session_state.clicked_pts = []
if "img_rotation" not in st.session_state:
    st.session_state.img_rotation = 0.0

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
def rotate_image(cv_img, angle_deg):
    """หมุนรูปภาพตามองศาที่กำหนดรอบจุดศูนย์กลางภาพ"""
    if abs(angle_deg) < 0.1:
        return cv_img

    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)

    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(
        cv_img, rot_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated_img

def calculate_theta_from_2points(pt1, pt2):
    """
    คำนวณมุม Theta (θ) จากจุด 2 จุดที่ผู้ใช้คลิก
    - pt1, pt2: (x, y)
    - มุมวัดจากแกน X ฝั่งขวา ทวนเข็มนาฬิกา (หน่วยองศา: 90° - 180°)
    """
    dx = pt2["x"] - pt1["x"]
    dy = pt2["y"] - pt1["y"]

    if dx == 0:
        return 90.0

    # สลับแกน Y เนื่องจากพิกัดรูปภาพ Y ชี้ลง
    # ความชัน m = dy / dx ในระบบพิกัดคาร์ทีเซียน
    m = -dy / dx

    phi_rad = math.atan(abs(m))
    phi_deg = math.degrees(phi_rad)

    # ปรับให้อยู่ในย่าน 90° - 180°
    if m >= 0:
        theta = 180.0 - phi_deg
    else:
        theta = 90.0 + phi_deg

    return max(90.0, min(180.0, theta))

def draw_hud(cv_img, points):
    """วาดจุดและเส้นเชื่อม 2 จุดบนภาพ"""
    img_out = cv_img.copy()

    # วาดจุดที่คลิก
    for i, p in enumerate(points):
        pt = (int(p["x"]), int(p["y"]))
        # จุดวงกลมแดงขอบเหลือง
        cv2.circle(img_out, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img_out, pt, 9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_out, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # ถ้าครบ 2 จุด ให้วาดเส้นลากเชื่อมต่อ และเส้น Baseline
    if len(points) == 2:
        p1, p2 = points[0], points[1]
        pt1 = (int(p1["x"]), int(p1["y"]))
        pt2 = (int(p2["x"]), int(p2["y"]))

        # 1. เส้นลากทับแนวเกลียว (สีแดง)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # 2. เส้นขนานพื้น Baseline (สีฟ้า)
        min_x = min(pt1[0], pt2[0]) - 50
        max_x = max(pt1[0], pt2[0]) + 50
        base_y = max(pt1[1], pt2[1])
        cv2.line(img_out, (min_x, base_y), (max_x, base_y), (255, 200, 0), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix ตามโมเดลที่เลือก"""
    if "Model 5-8-13" in model_name:
        ideal_deg = 155.0
        x = abs(theta - ideal_deg)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:
        ideal_deg = 136.0
        x = abs(theta - ideal_deg)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return brix, x, ideal_deg

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 1. เลือกโมเดลสับปะรด")
    model_choice = st.radio(
        "ระบุประเภทโมเดล:",
        options=[
            "Model 5-8-13 (มุมอุดมคติ 155°)",
            "Model 8-13-21 (มุมอุดมคติ 136°)"
        ]
    )

    st.markdown("---")
    st.header("💡 วิธีการใช้งาน")
    st.markdown("""
    1. **อัปโหลดภาพสับปะรด**
    2. ใช้ **Slider หมุนภาพ** เพื่อตั้งภาพให้ตรง
    3. **คลิกเลือกจุด 2 จุด** บนรูปภาพตามแนวเส้นเกลียว
    4. ระบบจะคำนวณมุมองศา ($\Theta$) และค่า Brix ให้ทันที
    """)

# -----------------------------------------------------------------------------
# 5. Main UI & App Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.3, 1])

    # โหลดรูปภาพ
    raw_pil_image = PIL.Image.open(uploaded_file).convert("RGB")
    cv_img_orig = cv2.cvtColor(np.array(raw_pil_image), cv2.COLOR_RGB2BGR)

    with col2:
        st.subheader("🖼️ 1. หมุนรูปภาพให้ตั้งตรง")
        
        # สไลเดอร์หมุนภาพ
        img_angle = st.slider(
            "หมุนปรับระดับภาพ (องศา):",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.img_rotation),
            step=0.5,
            format="%.1f°"
        )
        st.session_state.img_rotation = img_angle

        # หมุนภาพตามองศา slider
        rotated_cv_img = rotate_image(cv_img_orig, st.session_state.img_rotation)

        st.markdown("---")
        st.subheader("📍 2. จัดการจุดเล็งเกลียว (2 จุด)")

        # ปุ่มล้างจุด
        if st.button("🗑️ ล้างจุดที่เลือกทั้งหมด (Clear Points)"):
            st.session_state.clicked_pts = []
            st.rerun()

        # แสดงพิกัดจุดที่กด
        pts_count = len(st.session_state.clicked_pts)
        st.write(f"**สถานะการกดจุด:** เลือกแล้ว `{pts_count} / 2` จุด")

        if pts_count < 2:
            st.info("👉 คลิกบนรูปภาพฝั่งซ้ายเพื่อระบุจุดที่ 1 และจุดที่ 2 ตามแนวเกลียว")

        # ---------------------------------------------------------------------
        # คำนวณผลลัพธ์เมื่อกดครบ 2 จุด
        # ---------------------------------------------------------------------
        if pts_count == 2:
            p1, p2 = st.session_state.clicked_pts[0], st.session_state.clicked_pts[1]
            calculated_theta = calculate_theta_from_2points(p1, p2)
            brix_val, diff_x, ideal_angle = calc_brix(calculated_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ Brix")
            
            m1, m2 = st.columns(2)
            m1.metric("มุมเกลียวที่วัดได้ (θ)", f"{calculated_theta:.2f}°")
            m2.metric("🍬 ค่าความหวานประเมิน", f"{brix_val:.2f} °Brix")

            st.success(f"📍 มุมอุดมคติ: `{ideal_angle:.1f}°` | ค่าความเบี่ยงเบน ($x$): `{diff_x:.2f}°`")

    with col1:
        st.subheader("🎯 คลิกเล็งจุดบนรูปภาพ")

        # วาดภาพที่มีจุดและเส้นกด
        drawn_pil_img = draw_hud(rotated_cv_img, st.session_state.clicked_pts)

        # คอมโพเนนต์รับพิกัดการกด (Click coordinates listener)
        value = streamlit_image_coordinates(drawn_pil_img, key="pil_image")

        # รับค่าการคลิกใหม่
        if value is not None:
            new_pt = {"x": value["x"], "y": value["y"]}
            
            # ป้องกันการบันทึกจุดซ้ำตำแหน่งเดิมขณะ Rerun
            if len(st.session_state.clicked_pts) == 0 or (st.session_state.clicked_pts[-1] != new_pt):
                if len(st.session_state.clicked_pts) < 2:
                    st.session_state.clicked_pts.append(new_pt)
                    st.rerun()
