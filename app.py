import math
import os
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Pineapple Manual Measurement & Brix Analyzer", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียวสับปะรด (Manual 2-Point)")

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------
def rotate_image(cv_img, angle_deg):
    if abs(angle_deg) < 0.1:
        return cv_img
    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(cv_img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def calculate_theta_from_2points(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    if dx == 0:
        return 90.0
    m = -dy / dx
    phi_deg = math.degrees(math.atan(abs(m)))
    theta = (180.0 - phi_deg) if m >= 0 else (90.0 + phi_deg)
    return max(90.0, min(180.0, theta))

def calc_brix(theta, model_name):
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
# 2. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ เลือกโมเดลสับปะรด")
    model_choice = st.radio(
        "ระบุประเภทโมเดล:",
        options=["Model 5-8-13 (มุมอุดมคติ 155°)", "Model 8-13-21 (มุมอุดมคติ 136°)"]
    )

# -----------------------------------------------------------------------------
# 3. Main UI
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.3, 1])

    raw_pil_image = PIL.Image.open(uploaded_file).convert("RGB")
    cv_img_orig = cv2.cvtColor(np.array(raw_pil_image), cv2.COLOR_RGB2BGR)

    with col2:
        st.subheader("🖼️ 1. หมุนรูปภาพให้ตั้งตรง")
        img_angle = st.slider("หมุนปรับระดับภาพ (องศา):", -180.0, 180.0, 0.0, 0.5)
        
        # หมุนภาพ
        rotated_cv_img = rotate_image(cv_img_orig, img_angle)
        rotated_pil_img = PIL.Image.fromarray(cv2.cvtColor(rotated_cv_img, cv2.COLOR_BGR2RGB))

    with col1:
        st.subheader("🎯 คลิกเล็งจุด 2 จุดบนภาพ")
        st.caption("คลิกเลือกจุดที่ 1 และจุดที่ 2 บนแนวเกลียวสับปะรด")

        # ผ้าใบโต้ตอบสำหรับจิ้มจุด (Interactive Canvas)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=rotated_pil_img,
            update_streamlit=True,
            height=rotated_pil_img.height,
            width=rotated_pil_img.width,
            drawing_mode="point",
            key="canvas",
        )

    # ดึงพิกัดจาก Canvas เมื่อผู้ใช้จิ้มจุด
    points = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "circle":
                points.append((obj.get("left"), obj.get("top")))

    with col2:
        st.markdown("---")
        st.subheader("📊 ผลการคำนวณ Brix")
        st.write(f"**จำนวนจุดที่เลือก:** `{len(points)} / 2` จุด")

        if len(points) >= 2:
            p1, p2 = points[0], points[1]
            calculated_theta = calculate_theta_from_2points(p1, p2)
            brix_val, diff_x, ideal_angle = calc_brix(calculated_theta, model_choice)

            m1, m2 = st.columns(2)
            m1.metric("มุมเกลียวที่วัดได้ (θ)", f"{calculated_theta:.2f}°")
            m2.metric("🍬 ค่าความหวานประเมิน", f"{brix_val:.2f} °Brix")

            st.success(f"📍 มุมอุดมคติ: `{ideal_angle:.1f}°` | ค่าความเบี่ยงเบน ($x$): `{diff_x:.2f}°`")
        else:
            st.info("👉 คลิกเลือกจุดให้ครบ 2 จุดบนภาพเพื่อคำนวณค่า Brix")
