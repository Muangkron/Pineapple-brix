import streamlit as st
import cv2
import numpy as np

# =================================================================
# 1. ฟังก์ชันวิเคราะห์ตาสับปะรด (ที่มีระบบเซฟตี้ดัก Error บรรทัด 87)
# =================================================================
def analyze_pineapple(img):
    """
    ฟังก์ชันรับภาพเข้ามาประมวลผลแนวตาสับปะรด
    img: ภาพที่ส่งเข้ามาจะเป็นระบบสี BGR เหมือนใน Google Colab เป๊ะๆ
    """
    
    # 💥 [จุดที่ 1] แปะโค้ดประมวลผลภาพของน้องตรงนี้!
    # ให้น้องเอาโค้ดจาก Colab ส่วนที่ทำ cv2.inRange, คัดแยกสี, หา Contours 
    # จนถึงคำสั่งที่ใช้หาค่า vx, vy, x0, y0 มาใส่ตรงนี้ได้เลยครับ
    # (ตัวแปรภาพต้นฉบับให้ใช้คำว่า img ตามชื่อฟังก์ชันด้านบนนะ)
    
    # -------------------------------------------------------------
    # สมมติตัวอย่างโค้ดเบื้องหลังของน้อง (ลบหรือแก้ไขตามจริงได้เลยครับ)
    # เช่น: 
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower_color, upper_color)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ... โค้ดคำนวณหาค่า vx, vy, x0, y0 ...
    # -------------------------------------------------------------

    
    # 🛡️ [จุดที่ 2] ระบบเซฟตี้ป้องกันหน้าเว็บขึ้น TypeError กล่องแดง
    # ตรวจสอบก่อนว่าคำนวณหาพิกัดตาสับปะรดเจอไหม ถ้าไม่เจอ (เป็น None) ให้ดีดออกทันที ไม่ให้โค้ดพัง
    if 'vx' not in locals() or 'vy' not in locals() or vx is None or vy is None or x0 is None or y0 is None:
        return None, "ระบบตรวจไม่พบแนวร่องตาสับปะรดในภาพนี้ กรุณาเปลี่ยนใช้ภาพที่สว่างและเห็นตาเฉียงชัดเจนขึ้นครับ"


    # 📌 บรรทัดที่ 87 เจ้าปัญหาเดิม (ตอนนี้ปลอดภัยแล้วเพราะผ่านตัวกรองด้านบนมาได้)
    vx, vy, x0, y0 = map(float, [vx, vy, x0, y0])
    
    
    # 💥 [จุดที่ 3] โค้ดส่วนคำนวณ Brix และวาดเส้นแสดงผลของน้อง
    # เอาโค้ดส่วนที่คำนวณมุม คำนวณค่าความหวาน และสั่งวาดเส้น cv2.line มาใส่ต่อตรงนี้ครับ
    
    # สมมติตัวแปรผลลัพธ์สุดท้าย (แก้ไขตามโครงสร้างการ Return ค่าของน้องได้เลย)
    # ตัวอย่างเช่น ถ้าน้องต้องการส่งภาพผลลัพธ์กลับไปโชว์ ให้ใส่ในตัวแปร result_img
    result_img = img.copy() 
    # cv2.line(result_img, ...)
    
    # ส่งค่ากลับไปแสดงผลที่หน้าเว็บ (ส่งภาพที่วาดเส้นแล้ว และส่งข้อความ Error เป็น None)
    return result_img, None


# =================================================================
# 2. ส่วนแสดงผลหน้าเว็บ Streamlit (แก้ไขระบบรับภาพให้เป็น BGR)
# =================================================================
st.set_page_config(page_title="Pineapple Brix Detector", page_icon="🍍")
st.title("ระบบวิเคราะห์ความหวานจากตาสับปะรด 🍍")
st.write("โครงงานพัฒนาแอปพลิเคชันวิเคราะห์คุณภาพสับปะรด")

# ช่องอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("กรุณาอัปโหลดรูปภาพสับปะรด (แนวตั้งหรือเห็นตาสับปะรดชัดเจน)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # 🛠️ [จุดที่ 4] บังคับให้ Streamlit อ่านภาพออกมาเป็น BGR เป๊ะๆ เหมือน cv2.imread ใน Colab
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # สร้างคอลัมน์โชว์ภาพ เปรียบเทียบ ก่อน-หลัง
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 รูปภาพต้นฉบับ")
        # เวลา Streamlit จะโชว์รูป ต้องแปลงกลับเป็น RGB แป๊บหนึ่ง ไม่งั้นสีหน้าเว็บจะเพี้ยน
        st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("📊 ผลการวิเคราะห์")
        
        # ส่งภาพเข้าฟังก์ชันไปคำนวณ
        with st.spinner("กำลังประมวลผลภาพ..."):
            final_result, error_message = analyze_pineapple(opencv_img)
            
        # ตรวจสอบเงื่อนไขผลลัพธ์
        if error_message:
            # ถ้าหาตาสับปะรดไม่เจอ แทนที่จะขึ้นกล่องแดงระเบิด จะขึ้นคำเตือนสีส้มแนะนำผู้ใช้แทน ปลอดภัย 100%
            st.warning(error_message)
        else:
            # ถ้าคำนวณผ่านฉลุย ให้โชว์ภาพที่ประมวลผลแล้ว
            st.image(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success("ประมวลผลสำเร็จ!")
            
            # 💥 น้องสามารถเพิ่ม st.write() เพื่อแสดงค่าความหวาน Brix ที่คำนวณได้ตรงนี้เพิ่มเติมได้เลยครับ
