import streamlit as st
import cv2
import numpy as np

# ตั้งค่าหน้าเว็บสตรีมลิต
st.set_page_config(page_title="Pineapple Brix Analyzer", layout="centered")

st.title("ระบบวิเคราะห์ความหวานจากตาสับปะรด 🍍")
st.write("เวอร์ชันอัปเกรด: เพิ่มระบบกรองพื้นหลังและจุดรบกวน (Noise/Background Isolation)")
st.markdown("---")

# =========================================
# SELECT MODEL
# =========================================
model = st.selectbox("กรุณาเลือก Model ที่ต้องการใช้งาน:", ["model2", "model3"])

# =========================================
# UPLOAD IMAGE
# =========================================
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรดของคุณ (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # อ่านไฟล์รูปภาพจากหน้าเว็บเข้าสู่ระบบ OpenCV BGR
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # แปลงเป็น RGB ตามขั้นตอนแรกในโค้ดเดิมของน้อง
    rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    # =========================
    # Resize 
    # =========================
    h, w = rgb.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale)
        h, w = rgb.shape[:2]  # อัปเดตขนาดพิกัดหลังย่อรูปเพื่อใช้คำนวณขอบเขตสับปะรด

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    FORCE_LEFT_TO_RIGHT = True

    # =========================
    # PREPROCESS 
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=3,
        tileGridSize=(8,8)
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel
    )

    # =========================
    # FIND CONTOURS 
    # =========================
    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []

    # 🛡️ เกราะชั้นที่ 1: กำหนดระยะขอบปลอดภัย (Margin 3%) เพื่อตัดจุดรบกวนที่ติดขอบรูปภาพทิ้งไปทันที
    margin_w = w * 0.03
    margin_h = h * 0.03

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # คัดกรอง: หากพิกัดอยู่ในขอบเขตพื้นหลังรอบนอก ให้ข้ามไป ไม่นับเป็นตาสับปะรด
        if x < margin_w or y < margin_h or (x + cw) > (w - margin_w) or (y + ch) > (h - margin_h):
            continue

        ratio = cw / (ch + 1e-6)
        if ratio < 0.4 or ratio > 2.5:
            continue

        centers.append([
            x + cw/2,
            y + ch/2
        ])

    # ==================================================
    # CHECK & FILTER OUTLIERS (🛡️ เกราะชั้นที่ 2: ระบบคัดแยกพื้นหลังสถิติขั้นสูง)
    # ==================================================
    if len(centers) < 10:
        st.warning("พบตาสับปะรดน้อยเกินไป กรุณาเปลี่ยนรูปภาพถ่ายที่ชัดเจน หรือขยับให้ผลสับปะรดอยู่ตรงกลางรูปครับ")
    else:
        pts_all = np.array(centers, dtype=np.float32)
        
        # หาค่ามัธยฐาน (Median) เพื่อล็อกพิกัดจุดใจกลางของกลุ่มตาสับปะรดหลัก (ช่วยให้ไม่โดนพื้นหลังดึงตำแหน่งดิ่งพัง)
        med_x = np.median(pts_all[:, 0])
        med_y = np.median(pts_all[:, 1])
        
        # คำนวณระยะห่าง (Euclidean Distance) ของทุกจุดเทียบกับจุดใจกลางสับปะรด
        dists = np.sqrt((pts_all[:, 0] - med_x)**2 + (pts_all[:, 1] - med_y)**2)
        
        # กำหนดเกณฑ์คัดทิ้งโดยอิงหลักสถิติรังวัดกลุ่มข้อมูล (Mean + 1.5 * Standard Deviation)
        # จุดไหนที่ลอยอยู่เดี่ยวๆ โดดเดี่ยวบนพื้นหลัง จะมีระยะทางเกินเกณฑ์นี้และถูกโยนทิ้งไป
        threshold_dist = np.mean(dists) + 1.5 * np.std(dists)
        
        filtered_centers = []
        for i, c in enumerate(centers):
            if dists[i] <= threshold_dist:
                filtered_centers.append(c)

        # ตรวจสอบซ้ำอีกครั้งหลังจากคัดแยก Noise ออกไปแล้ว
        if len(filtered_centers) < 10:
            st.warning("ระบบสกัดจุดรบกวนแล้วพบตาสับปะรดหลักไม่เพียงพอ กรุณาใช้รูปถ่ายที่ซูมเห็นสับปะรดชัดขึ้นครับ")
        else:
            pts = np.array(filtered_centers, dtype=np.float32)

            # เรียงจากซ้ายบน -> ขวาล่าง
            pts = pts[np.argsort(
                pts[:,0] + pts[:,1]
            )]

            # ใช้ประมาณ 50 จุดแรก
            pts = pts[:50]

            if not FORCE_LEFT_TO_RIGHT:
                pts[:,0] = -pts[:,0]

            # =========================
            # DRAW CONNECTIONS 
            # =========================
            for i in range(len(pts)-1):
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                cv2.line(rgb, p1, p2, (255,255,0), 2)

            # =========================
            # FIT MAIN SPIRAL (🛡️ เกราะชั้นที่ 3: ปรับเป็นแบบคำนวณทนทานต่อ Outliers)
            # =========================
            # เปลี่ยนพารามิเตอร์ระยะทางเป็น cv2.DIST_HUBER เพื่อป้องกันไม่ให้เศษกระจายตัวมาดึงทิศทางเส้นหลัก
            vx, vy, x0, y0 = cv2.fitLine(
                pts,
                cv2.DIST_HUBER,
                0,
                0.01,
                0.01
            )

            vx = float(vx.item())
            vy = float(vy.item())
            x0 = float(x0.item())
            y0 = float(y0.item())

            angle_main = np.degrees(
                np.arctan2(vy, vx)
            )

            if angle_main < 0:
                angle_main += 180

            L = 3000
            brix = None
            angle_used = None

            # =========================
            # MODEL2 
            # =========================
            if model == "model2":
                cv2.line(
                    rgb,
                    (int(x0-vx*L), int(y0-vy*L)),
                    (int(x0+vx*L), int(y0+vy*L)),
                    (255,0,0),
                    6
                )

                theta = np.radians(75)

                vx_r = (
                    vx*np.cos(theta)
                    - vy*np.sin(theta)
                )
                vy_r = (
                    vx*np.sin(theta)
                    + vy*np.cos(theta)
                )

                cv2.line(
                    rgb,
                    (int(x0-vx_r*L), int(y0-vy_r*L)),
                    (int(x0+vx_r*L), int(y0+vy_r*L)),
                    (0,0,255),
                    5
                )

                angle_rot = np.degrees(
                    np.arctan2(vy_r, vx_r)
                )

                if angle_rot < 0:
                    angle_rot += 180

                angle_used = angle_rot

                # ===== BRIX MODEL2 =====
                x_brix = abs(angle_main - 146)
                brix = (
                    0.0428*(x_brix**2)
                    - 0.9296*x_brix
                    + 16.037
                )

            # =========================
            # MODEL3 
            # =========================
            elif model == "model3":
                cv2.line(
                    rgb,
                    (int(x0-vx*L), int(y0-vy*L)),
                    (int(x0+vx*L), int(y0+vy*L)),
                    (255,0,0),
                    6
                )

                theta = np.radians(98)

                vx_r = (
                    vx*np.cos(theta)
                    - vy*np.sin(theta)
                )
                vy_r = (
                    vx*np.sin(theta)
                    + vy*np.cos(theta)
                )

                cv2.line(
                    rgb,
                    (int(x0-vx_r*L), int(y0-vy_r*L)),
                    (int(x0+vx_r*L), int(y0+vy_r*L)),
                    (0,0,255),
                    6
                )

                angle_98 = np.degrees(
                    np.arctan2(vy_r, vx_r)
                )

                if angle_98 < 0:
                    angle_98 += 180

                angle_used = angle_98

                # เส้นแนวระดับ
                cv2.line(
                    rgb,
                    (int(x0-L), int(y0)),
                    (int(x0+L), int(y0)),
                    (0,255,255),
                    3
                )

                # ===== BRIX MODEL3 =====
                x_brix = abs(angle_main - 135)
                brix = (
                    0.0366*(x_brix**2)
                    - 0.8924*x_brix
                    + 16.696
                )

            # =========================
            # DRAW POINTS 
            # =========================
            for p in pts:
                cv2.circle(
                    rgb,
                    (int(p[0]), int(p[1])),
                    5,
                    (0,255,0),
                    -1
                )

            # =========================
            # SHOW BRIX ON IMAGE & WEB RESULT
            # =========================
            if brix is not None:
                cv2.putText(
                    rgb,
                    f"Brix = {brix:.2f}%",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255,0,255),
                    3
                )

            cv2.putText(
                rgb,
                f"Main={angle_main:.1f}  Brix={brix:.2f}%",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 255),
                3
            )

            # แสดงภาพผลลัพธ์ผ่านหน้าเว็บสตรีมลิต
            st.image(rgb, caption=f"ผลการวิเคราะห์สับปะรดด้วย {model}", use_container_width=True)

            # รายงานสรุปข้อมูลตัวเลขโมเดลคณิตศาสตร์
            st.markdown("### 📊 ผลสรุปตัวเลขจากการคำนวณ")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Main Spiral Angle", f"{angle_main:.2f}°")
            with c2:
                st.metric("Angle Used", f"{angle_used:.2f}°")
            with c3:
                st.metric("Predicted Brix", f"{brix:.2f}%")
