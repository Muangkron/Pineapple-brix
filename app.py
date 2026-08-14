import math
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Pineapple Spiral Brix", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียว & คำนวณ Brix")

if "clicked_pts" not in st.session_state:
    st.session_state.clicked_pts = []
if "img_rotation" not in st.session_state:
    st.session_state.img_rotation = 0.0

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def rotate_image(cv_img, angle_deg):
    if abs(angle_deg) < 0.1:
        return cv_img
    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(cv_img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

# -----------------------------------------------------------------------------
# ฟังก์ชันคำนวณมุมใหม่ (แก้ไขปัญหาการวัดสลับด้าน)
# -----------------------------------------------------------------------------
def calculate_theta_measured(p1, p2):
    """
    คำนวณมุมเกลียวสับปะรดเทียบกับแกน X ทางขวา
    p1, p2: จุด 2 จุดที่คลิกเลือกบนเส้นเกลียว
    """
    dx = p2["x"] - p1["x"]
    dy = -(p2["y"] - p1["y"])  # กลับทิศ Y เพราะพิกัดจอภาพ Y ชี้ลงล่าง

    if dx == 0:
        return 90.0

    # คำนวณมุมองศาของเส้นตรงเทียบกับแกน X ขวา
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # ปรับให้อยู่ในมุมป้าน/แหลม เทียบกับแกนนอนขวา (0 - 180 องศา)
    if angle_deg < 0:
        angle_deg += 180.0

    return angle_deg

def draw_measurement_hud(cv_img, points):
    img_out = cv_img.copy()
    
    # วาดจุดที่คลิก
    for i, p in enumerate(points):
        pt = (int(p["x"]), int(p["y"]))
        cv2.circle(img_out, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img_out, pt, 9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_out, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if len(points) == 2:
        pt1 = (int(points[0]["x"]), int(points[0]["y"]))
        pt2 = (int(points[1]["x"]), int(points[1]["y"]))

        # คำนวณมุม
        theta = calculate_theta_measured(points[0], points[1])

        # หาจุดศูนย์กลางระหว่าง 2 จุดเพื่อเป็นจุดตัดแกนนอน
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        center_pt = (mid_x, mid_y)

        # วาดเส้นแกน X อ้างอิงผ่านจุดกลาง (เส้นสีแดงแนวนอนแบบในรูปของคุณ)
        axis_len = 150
        cv2.line(img_out, (mid_x - axis_len, mid_y), (mid_x + axis_len, mid_y), (0, 0, 255), 2, cv2.LINE_AA)

        # วาดเส้นเกลียวสับปะรดเฉียง (เส้นสีแดง)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # วาดส่วนโค้งบอกมุม (สีเขียว) วัดจากแกน X ขวาไปหาเส้นเกลียว
        arc_radius = 40
        cv2.ellipse(img_out, center_pt, (arc_radius, arc_radius), 0, 0, -int(theta), (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงข้อความค่ามุมบนรูป
        cv2.putText(img_out, f"{theta:.1f} deg", (mid_x + 50, mid_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    # สมการคำนวณ Brix
    if "Model 5-8-13" in model_name:
        ideal_deg = 155.0
        x = abs(theta - ideal_deg)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:
        ideal_deg = 136.0
        x = abs(theta - ideal_deg)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return brix

# -----------------------------------------------------------------------------
# Sidebar & Main UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ เลือกโมเดลสับปะรด")
    model_choice = st.radio(
        "ระบุประเภทโมเดล:",
        options=["Model 5-8-13 (มุมอุดมคติ 155°)", "Model 8-13-21 (มุมอุดมคติ 136°)"]
    )

uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.3, 1])

    raw_pil_image = PIL.Image.open(uploaded_file).convert("RGB")
    cv_img_orig = cv2.cvtColor(np.array(raw_pil_image), cv2.COLOR_RGB2BGR)

    with col2:
        st.subheader("🖼️ 1. หมุนรูปภาพ (ถ้าจำเป็น)")
        img_angle = st.slider("หมุนปรับระดับภาพ (องศา):", -180.0, 180.0, float(st.session_state.img_rotation), 0.5)
        st.session_state.img_rotation = img_angle
        rotated_cv_img = rotate_image(cv_img_orig, st.session_state.img_rotation)

        st.markdown("---")
        st.subheader("📍 2. จัดการจุดเล็งเกลียว (2 จุด)")
        if st.button("🗑️ ล้างจุดทั้งหมด"):
            st.session_state.clicked_pts = []
            st.rerun()

        pts_count = len(st.session_state.clicked_pts)
        st.write(f"เลือกแล้ว: `{pts_count} / 2` จุด")

        if pts_count == 2:
            calculated_theta = calculate_theta_measured(st.session_state.clicked_pts[0], st.session_state.clicked_pts[1])
            brix_val = calc_brix(calculated_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ")
            st.metric("มุมเกลียวที่วัดได้ (θ)", f"{calculated_theta:.1f}°")
            st.metric("🍬 ค่า Brix คำนวณได้", f"{brix_val:.2f}")

    with col1:
        st.subheader("🎯 คลิกเล็งจุดบนรูปภาพ")
        drawn_pil_img = draw_measurement_hud(rotated_cv_img, st.session_state.clicked_pts)
        
        value = streamlit_image_coordinates(drawn_pil_img, key="pil_image")

        if value is not None:
            new_pt = {"x": value["x"], "y": value["y"]}
            if len(st.session_state.clicked_pts) == 0 or (st.session_state.clicked_pts[-1] != new_pt):
                if len(st.session_state.clicked_pts) < 2:
                    st.session_state.clicked_pts.append(new_pt)
                    st.rerun()
