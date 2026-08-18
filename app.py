import math
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------------------------
# 1. Config & State
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pineapple Spiral Brix", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียวสับปะรด & คำนวณ Brix")

if "points" not in st.session_state:
    st.session_state.points = []
if "angle_rotation" not in st.session_state:
    st.session_state.angle_rotation = 0.0

# -----------------------------------------------------------------------------
# 2. ฟังก์ชันคำนวณมุมป้านจากแกนนอนขวา (เขียนใหม่แบบตรงไปตรงมา)
# -----------------------------------------------------------------------------
def calculate_spiral_angle(pt1, pt2):
    """
    คำนวณมุมป้านทวนเข็มนาฬิกาจากแกนนอนขวา (0 - 180 องศา)
    """
    dx = abs(pt2["x"] - pt1["x"])
    dy = abs(pt2["y"] - pt1["y"])

    if dx == 0:
        return 90.0

    # 1. หามุมแหลมระหว่างเส้นตรงกับแกนนอน (0 - 90 องศา)
    phi = math.degrees(math.atan(dy / dx))

    # 2. แปลงเป็นมุมป้านฝั่งแกน X ขวาชี้ขึ้นไปหาเส้นเกลียว (180 - phi)
    theta = 180.0 - phi
    return theta

def rotate_img(cv_img, angle):
    if abs(angle) < 0.1:
        return cv_img
    h, w = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def draw_hud(cv_img, pts):
    img = cv_img.copy()
    
    # วาดจุดเล็ง
    for i, p in enumerate(pts):
        pos = (int(p["x"]), int(p["y"]))
        cv2.circle(img, pos, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pos, 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"P{i+1}", (pos[0] + 10, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(pts) == 2:
        p1, p2 = (int(pts[0]["x"]), int(pts[0]["y"])), (int(pts[1]["x"]), int(pts[1]["y"]))
        
        # คำนวณมุม
        theta = calculate_spiral_angle(pts[0], pts[1])
        
        # วาดเส้นเกลียวสีแดง
        cv2.line(img, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)

        # วาดเส้นแกนนอนสีแดงตัดผ่านจุดกึ่งกลาง (แบบในรูปของคุณ)
        mid_x, mid_y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
        cv2.line(img, (mid_x - 150, mid_y), (mid_x + 150, mid_y), (0, 0, 255), 2, cv2.LINE_AA)

        # ข้อความแสดงมุมองศา
        cv2.putText(img, f"Theta: {theta:.1f} deg", (mid_x + 20, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def calc_brix_val(theta, model_type):
    if "Model 5-8-13" in model_type:
        ideal = 146.0
        diff = abs(theta - ideal)
        brix = (-0.0196 * (diff**2)) + (0.0045 * diff) + 16.757
    else:
        ideal = 135.0
        diff = abs(theta - ideal)
        brix = (0.0082 * (diff**2)) - (0.6667 * diff) + 16.362
    return brix

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ตัวเลือกโมเดล")
    model_choice = st.radio(
        "เลือกรุ่นโมเดลสับปะรด:",
        options=["Model 5-8-13 (มุมอุดมคติ 146°)", "Model 8-13-21 (มุมอุดมคติ 135°)"]
    )

# -----------------------------------------------------------------------------
# 4. Main UI
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_left, col_right = st.columns([1.3, 1])

    raw_pil = PIL.Image.open(uploaded_file).convert("RGB")
    cv_img_raw = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)

    with col_right:
        st.subheader("🖼️ 1. ปรับหมุนภาพ")
        rot_val = st.slider("องศาหมุนภาพ:", -180.0, 180.0, float(st.session_state.angle_rotation), 0.5)
        st.session_state.angle_rotation = rot_val
        cv_img_rotated = rotate_img(cv_img_raw, st.session_state.angle_rotation)

        st.markdown("---")
        st.subheader("📍 2. ควบคุมการเล็งจุด")
        if st.button("🗑️ ล้างจุดทั้งหมด (Clear)"):
            st.session_state.points = []
            st.rerun()

        st.write(f"สถานะ: เลือกแล้ว `{len(st.session_state.points)} / 2` จุด")

        if len(st.session_state.points) == 2:
            calc_theta = calculate_spiral_angle(st.session_state.points[0], st.session_state.points[1])
            brix_res = calc_brix_val(calc_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลลัพธ์")
            st.metric("มุมเกลียวที่วัดได้ (θ)", f"{calc_theta:.1f}°")
            st.metric("ค่าความหวานประเมิน (Brix)", f"{brix_res:.2f} °Brix")

    with col_left:
        st.subheader("🎯 คลิกเลือกจุด 2 จุดบนเกลียวสับปะรด")
        drawn_image = draw_hud(cv_img_rotated, st.session_state.points)
        
        clicked_val = streamlit_image_coordinates(drawn_image, key="img_coords")

        if clicked_val:
            pt_new = {"x": clicked_val["x"], "y": clicked_val["y"]}
            if len(st.session_state.points) == 0 or (st.session_state.points[-1] != pt_new):
                if len(st.session_state.points) < 2:
                    st.session_state.points.append(pt_new)
                    st.rerun()
