import math
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ล็อคเวอร์ชัน Streamlit และเปลี่ยน Python เป็น 3.11 ในหน้า Dashboard ตามที่เคยแนะนำ
# เพื่อให้ streamlit_image_coordinates ทำงานได้

st.set_page_config(page_title="Pineapple Spiral Brix", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียว (ทวนเข็มจากแกน X ขวา) & คำนวณ Brix")

if "clicked_pts" not in st.session_state:
    st.session_state.clicked_pts = []
if "img_rotation" not in st.session_state:
    st.session_state.img_rotation = 0.0

# -----------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------
def rotate_image(cv_img, angle_deg):
    if abs(angle_deg) < 0.1:
        return cv_img
    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(cv_img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def get_p1_p2(pts):
    """เรียงจุดให้ P1 อยู่ด้านล่าง (Y มากสุด) เสมอ เพื่อเป็นจุดหมุน"""
    if pts[0]["y"] > pts[1]["y"]:
        return pts[0], pts[1]
    else:
        return pts[1], pts[0]

def calculate_theta_measured(p1, p2):
    """
    คำนวณมุมทวนเข็มนาฬิกาจากแกน X ด้านขวาขึ้นไปหาเส้น P1-P2
    p1: จุดหมุน (อยู่ด้านล่าง), p2: จุดปลายเส้น (อยู่ด้านบน)
    """
    dx = p2["x"] - p1["x"]
    dy = -(p2["y"] - p1["y"])  # กลับทิศ Y เพราะพิกัดภาพ Y ชี้ลงล่าง

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360.0  # ปรับให้เป็นบวกทวนเข็มนาฬิกา 0-360

    return angle_deg

def draw_measurement_hud(cv_img, points):
    img_out = cv_img.copy()
    for i, p in enumerate(points):
        pt = (int(p["x"]), int(p["y"]))
        cv2.circle(img_out, pt, 7, (0, 255, 255), -1, cv2.LINE_AA) # จุดกลางเหลือง
        cv2.circle(img_out, pt, 9, (0, 0, 255), 2, cv2.LINE_AA) # ขอบแดง
        cv2.putText(img_out, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if len(points) == 2:
        # 1. หาจุด P1 (ล่าง) และ P2 (บน)
        p1_coord, p2_coord = get_p1_p2(points)
        pt1 = (int(p1_coord["x"]), int(p1_coord["y"]))
        pt2 = (int(p2_coord["x"]), int(p2_coord["y"]))

        # 2. คำนวณมุม
        theta = calculate_theta_measured(p1_coord, p2_coord)

        # --- วาด HUD แสดงวิธีการวัด ---

        # ก. วาดแกน X อ้างอิงไปทางขวาจาก P1 (สีเหลือง)
        axis_len = 120
        axis_end = (pt1[0] + axis_len, pt1[1])
        cv2.line(img_out, pt1, axis_end, (0, 255, 255), 2, cv2.LINE_AA)
        
        # ใส่หัวลูกศรแกน X
        cv2.line(img_out, axis_end, (axis_end[0]-10, axis_end[1]-5), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img_out, axis_end, (axis_end[0]-10, axis_end[1]+5), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_out, "X (0 deg)", (axis_end[0] + 5, axis_end[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ข. วาดเส้นเกลียวที่วัดจริง P1-P2 (สีแดง)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # ค. วาดส่วนโค้งแสดงมุม (Angle Arc) (สีเขียว)
        arc_radius = 50
        # drawEllipse มุมเริ่มต้นคือแกน X ขวา ทวนเข็มนาฬิกา
        # startAngle=0 (แกน X ขวา), endAngle = -theta (เพราะ OpenCV drawEllipse ทิศทางตามเข็มเป็นบวก)
        cv2.ellipse(img_out, pt1, (arc_radius, arc_radius), 0, 0, -int(theta), (0, 255, 0), 2, cv2.LINE_AA)
        
        # ใส่ข้อความมุมตรงกลางเส้นเกลียว
        mid_pt = (int((pt1[0]+pt2[0])/2) + 15, int((pt1[1]+pt2[1])/2) - 15)
        cv2.putText(img_out, f"Measured: {theta:.1f} deg", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    # สมมติสูตร Brix ตามองศา (คุณสามารถปรับแก้สูตรจริงได้ที่นี่)
    # ตัวอย่าง: Brix = slope * theta + intercept
    if "Model 5-8-13" in model_name:
        # ตัวอย่างสูตร 1
        brix = 0.05 * theta + 8.0 
    else:
        # ตัวอย่างสูตร 2
        brix = -0.03 * theta + 20.0
    return brix

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    model_choice = st.radio(
        "เลือกโมเดลคำนวณ Brix:",
        options=["Model 5-8-13", "Model 8-13-21 (ตัวอย่าง)"]
    )
    st.markdown("---")
    st.markdown("""
    **วิธีวัด:**
    1. คลิก **จุดที่ 1** และ **จุดที่ 2** บนแนวเกลียวสับปะรด (คลิกจุดไหนก่อนก็ได้)
    2. ระบบจะสร้างจุดหมุนที่จุดด้านล่าง และวัดมุมทวนเข็มนาฬิกาจากแกน X ด้านขวาขึ้นไป
    """)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.3, 1])

    raw_pil_image = PIL.Image.open(uploaded_file).convert("RGB")
    cv_img_orig = cv2.cvtColor(np.array(raw_pil_image), cv2.COLOR_RGB2BGR)

    with col2:
        st.subheader("🖼️ 1. หมุนรูปภาพ (ถ้าจำเป็น)")
        img_angle = st.slider("หมุนภาพ (องศา):", -180.0, 180.0, float(st.session_state.img_rotation), 0.5)
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
            p1_coord, p2_coord = get_p1_p2(st.session_state.clicked_pts)
            calculated_theta = calculate_theta_measured(p1_coord, p2_coord)
            brix_val = calc_brix(calculated_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ")
            st.metric("มุมเกลียวที่วัดได้ (θ)", f"{calculated_theta:.1f}°")
            st.metric("🍬 ค่า Brix คำนวณได้", f"{brix_val:.2f}")

    with col1:
        st.subheader("🎯 คลิกเล็งจุดบนรูปภาพ")
        
        # วาดรูปที่มี HUD แสดงวิธีวัดมุม
        drawn_pil_img = draw_measurement_hud(rotated_cv_img, st.session_state.clicked_pts)
        
        # คอมโพเนนต์รับพิกัดคลิก
        value = streamlit_image_coordinates(drawn_pil_img, key="pil_image")

        if value is not None:
            new_pt = {"x": value["x"], "y": value["y"]}
            # ป้องกัน Rerun แล้วได้จุดเดิมซ้ำ
            if len(st.session_state.clicked_pts) == 0 or (st.session_state.clicked_pts[-1] != new_pt):
                if len(st.session_state.clicked_pts) < 2:
                    st.session_state.clicked_pts.append(new_pt)
                    st.rerun()
