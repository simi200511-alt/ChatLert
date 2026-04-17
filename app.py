import streamlit as st
import re
import os
from PIL import Image
from pythainlp import word_tokenize

# --- 1. ตั้งค่าหน้าจอและรูปภาพประกอบ (เอาไอคอนออก) ---
st.set_page_config(page_title="ChatLert", layout="centered")

# --- 🎨 โค้ด CSS Makeover ธีมสีม่วงอ่อน (Light Purple Theme) ---
st.markdown("""
    <style>
    /* เปลี่ยนสีพื้นหลังหลักของหน้าเว็บเป็นสีม่วงพาสเทลอ่อนๆ */
    .stApp {
        background-color: #fbf8ff;
    }

    /* ปรับสีของหัวข้อ (Header) ให้เป็นสีม่วงเข้ม */
    h1, h2, h3 {
        color: #6a1b9a !important;
    }

    /* ปรับแต่งกล่อง Metric (เจตนา, สกัดข้อมูล) ให้เป็นการ์ดดูพรีเมียม */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(123, 31, 162, 0.08);
        border-top: 4px solid #ba68c8;
    }

    /* ปรับสีและฟอนต์ของข้อความในกล่อง Metric */
    div[data-testid="stMetricLabel"] p {
        font-size: 15px !important;
        font-weight: bold !important;
        color: #8e24aa !important;
    }
    
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        font-size: 18px !important;
        line-height: 1.3 !important;
        color: #4a148c !important;
    }

    /* ปรับปุ่มกดให้เป็นโทนม่วง */
    .stButton>button {
        background-color: #ab47bc !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0px 4px 6px rgba(171, 71, 188, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #8e24aa !important;
        box-shadow: 0px 6px 12px rgba(142, 36, 170, 0.4) !important;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

ASSETS_DIR = "assets"
def load_image_asset(file_name):
    path = os.path.join(ASSETS_DIR, file_name)
    if os.path.exists(path):
        return Image.open(path)
    return None

img_logo = load_image_asset("logo.png")
img_bot = load_image_asset("bot.png")
img_user = load_image_asset("user.png")

# --- 2. ตัวประมวลผล ChatLert Engine (NLP Pipeline) ---
class ChatLertEngine:
    def __init__(self):
        self.junk_tokens = {"อะ", "น้า", "นะฮะ", "เบย", " ", "นะ"}

    def reduce_repeated(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def process(self, raw_text):
        female_suffixes = ["ค่ะ", "คะ", "ขา", "ค่า", "คร่า", "คระ"]
        male_suffixes = ["ครับ", "งับ", "ค้าบ", "คั้บ", "คับ", "คาบ"]
        
        has_polite_suffix = any(s in raw_text for s in female_suffixes + male_suffixes)
        is_female = any(s in raw_text for s in female_suffixes)

        step1 = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', raw_text)
        step2 = self.reduce_repeated(step1)
        
        original_is_question = any(word in step2 for word in ["ไหม", "อะไร", "ไหน", "ยัง", "เมื่อไหร่", "กี่", "มั้ย", "มัย", "หนาย", "ไร", "บ้าง", "ยาง"])

        step3 = step2
        correct_kha_step3 = "คะ" if original_is_question else "ค่ะ"
        for slang_kha in ["คระ", "ค่า", "คร่า"]:
            if slang_kha in step3:
                step3 = step3.replace(slang_kha, correct_kha_step3)
                
        slang_replacements = [
            ("หนายย", "ไหน"), ("หนาย", "ไหน"),
            ("อยุ่", "อยู่"), ("อยุ", "อยู่"), ("ยุ", "อยู่"),
            ("มั้ย", "ไหม"), ("มัย", "ไหม"),
            ("เด่ว", "เดี๋ยว"), ("เดว", "เดี๋ยว"),
            ("สนจัย", "สนใจ"),
            ("งับ", "ครับ"), ("ค้าบ", "ครับ"), ("คั้บ", "ครับ"), ("คาบ", "ครับ"), ("คับ", "ครับ"),
            ("ไร", "อะไร"),
            ("ถึงยาง", "ถึงยัง"), ("ส่งยาง", "ส่งยัง"), ("ได้ยาง", "ได้ยัง") 
        ]
        
        for slang, formal in slang_replacements:
            step3 = step3.replace(slang, formal)
            
        step3 = step3.replace("ออะไร", "อะไร")
        step3 = step3.replace("ออยู่", "อยู่")

        text_for_tokenize = step3.replace("ไหนคะ", "ไหน คะ").replace("ไหมคะ", "ไหม คะ")
            
        raw_tokens = word_tokenize(text_for_tokenize, engine="newmm")
        
        step5_tokens = [t.strip() for t in raw_tokens if t not in self.junk_tokens and t.strip() != ""]
        step5_text = "".join(step5_tokens)

        intent = "ข้อความทั่วไป"
        adjusted = step5_text

        if any(word in step5_text for word in ["ส่ง", "ถึงไหน", "ได้ของ", "พัสดุ", "วันไหน", "ถึงยัง", "ส่งยัง"]):
            intent = "สอบถามสถานะสินค้า"
            if "วัน" in step5_text:
                adjusted = "สินค้าจะจัดส่งถึงในวันไหน"
            elif "ถึงไหน" in step5_text:
                adjusted = "สินค้าถึงขั้นตอนไหนของการจัดส่งแล้ว"
            elif "ได้ของ" in step5_text:
                adjusted = "ได้รับสินค้าแล้วหรือยัง"
            else:
                adjusted = "สินค้าจัดส่งแล้วหรือยัง"
                
        elif any(word in step5_text for word in ["โอน", "จ่าย", "ยอด", "สลิป"]):
            intent = "แจ้งชำระเงิน"
            if "เช็ค" in step5_text or "ตรวจสอบ" in step5_text:
                adjusted = "โอนเงินเรียบร้อยแล้ว ตรวจสอบให้ด้วย"
            else:
                adjusted = "โอนเงินเรียบร้อยแล้ว"
                
        elif "เปลี่ยน" in step5_text or "เคลม" in step5_text:
            intent = "ขอเปลี่ยนสินค้า"
            if any(word in step5_text for word in ["ไซส์", "เล็ก", "ใหญ่", "ขนาด"]):
                adjusted = "ต้องการขอเปลี่ยนไซส์สินค้า"
            else:
                adjusted = "ต้องการขอเปลี่ยนสินค้า"
                
        elif any(word in step5_text for word in ["สนใจ", "ราคา", "สี", "ขนาด", "ไซส์", "เท่าไหร่", "กี่บาท"]):
            intent = "สอบถามรายละเอียดสินค้า"
            adjusted = step5_text

        info_product = "-"
        info_attr = "-"
        if "เสื้อ" in step5_text: info_product = "เสื้อ"
        if "กางเกง" in step5_text: info_product = "กางเกง"
        
        if "สี" in step5_text: info_attr = "สี (Color)"
        elif "ไซส์" in step5_text or "ขนาด" in step5_text: info_attr = "ไซส์ (Size)"
        elif "ราคา" in step5_text or "เท่าไหร่" in step5_text: info_attr = "ราคา (Price)"

        if has_polite_suffix:
            final_is_question = any(word in adjusted for word in ["ไหน", "ยัง", "ไหม", "อะไร", "เท่าไหร่", "กี่", "บ้าง"])
            if is_female:
                polite_word = "คะ" if final_is_question else "ค่ะ"
            else:
                polite_word = "ครับ"
            
            base_sentence = adjusted.rstrip("ครับคะค่ะ")
            final_output = f"{base_sentence}{polite_word}"
        else:
            final_output = adjusted

        return {
            "step_5_1": step1,
            "step_5_2": step2,
            "step_5_3": step3,
            "step_5_4": raw_tokens,
            "step_5_5": step5_tokens,
            "step_5_6": adjusted,
            "step_5_7": final_output,
            "step_5_8": intent,
            "product": info_product, "attr": info_attr, "has_suffix": has_polite_suffix
        }

# --- 3. ส่วนการแสดงผล (UI) รูปแบบแชท ---
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_l, col_r = st.columns([1, 6])
    with col_l:
        if img_logo: st.image(img_logo, width=80)
    with col_r:
        st.title("ChatLert NLP System")
        st.write("ระบบแปลงภาษาแชทลูกค้าเป็นภาษาไทยมาตรฐาน (Official Application)")
        st.caption("ผู้จัดทำ: นางสาวสิมิลันนา พาล์เมียรี่ ชไวเกิร์ต")
    
    st.divider()

    with st.expander("ข้อความตัวอย่าง (คลิกเพื่อคัดลอก)"):
        example_texts = """สินค้าถึงวันหนายยยค่าา\nโอนแล้วน้าา เช็คให้ทีงับ\nเสื้อตัวนี้ไซส์เล็กไป ขอเปลี่ยนหน่อยคระ\nสินค้าถึงยางงง\nสนจัยตัวนี้อยุ่ มีสีไรบ้างคะะะ"""
        st.code(example_texts, language="text")

    # --- แสดงประวัติแชท ---
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar=img_user):
                st.write(msg["content"])
        else:
            res = msg["content"]
            with st.chat_message("assistant", avatar=img_bot):
                
                custom_success_box = f"""
                <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                            border-left: 5px solid #ab47bc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 15px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <span style="font-size: 14px; color: #6a1b9a;"><b>ภาษามาตรฐาน:</b></span><br>
                    <span style="font-size: 20px; color: #4a148c; font-weight: bold;">{res['step_5_7']}</span>
                </div>
                """
                st.markdown(custom_success_box, unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns([1.2, 1, 1])
                with col_m1:
                    st.metric("เจตนา (Intent)", res["step_5_8"])
                with col_m2:
                    extracted_info = f"{res['product']} | {res['attr']}" if res['product'] != "-" or res['attr'] != "-" else "ไม่พบข้อมูลเฉพาะ"
                    st.metric("การสกัดข้อมูล", extracted_info)
                with col_m3:
                    suffix_status = "ปรับตามหางเสียง" if res["has_suffix"] else "ไม่ใช้หางเสียง"
                    st.metric("รูปแบบประโยค", suffix_status)

                with st.expander("ดูรายละเอียด NLP Pipeline 8 ขั้นตอน (Technical Details)"):
                    st.write(f"**1. การทำความสะอาด (Clean):** `{res['step_5_1']}`")
                    st.write(f"**2. การลดตัวอักษรซ้ำ (Reduced):** `{res['step_5_2']}`")
                    st.write(f"**3. การปรับคำแสลง (Slang Normalization):** `{res['step_5_3']}`")
                    st.write(f"**4. การตัดคำ (Word Tokenization):** `{res['step_5_4']}`")
                    st.write(f"**5. การลบคำที่ไม่จำเป็น (Token Cleaning):** `{res['step_5_5']}`")
                    st.write(f"**6. การปรับโครงสร้างประโยค (Sentence Adjustment):** `{res['step_5_6']}`")
                    st.write(f"**7. การเพิ่มคำสุภาพ (Add Polite Particle):** `{res['step_5_7']}`")
                    st.write(f"**8. การจำแนกเจตนา (Intent Classification):** `{res['step_5_8']}`")

    # --- รับข้อความเข้าทางช่องแชท (Chat Input) ---
    if prompt := st.chat_input("พิมพ์ข้อความแชทลูกค้าที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=img_user):
            st.write(prompt)

        engine = ChatLertEngine()
        res = engine.process(prompt)

        with st.chat_message("assistant", avatar=img_bot):
            
            custom_success_box = f"""
            <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        border-left: 5px solid #ab47bc; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 15px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <span style="font-size: 14px; color: #6a1b9a;"><b>ภาษามาตรฐาน:</b></span><br>
                <span style="font-size: 20px; color: #4a148c; font-weight: bold;">{res['step_5_7']}</span>
            </div>
            """
            st.markdown(custom_success_box, unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3 = st.columns([1.2, 1, 1])
            with col_m1:
                st.metric("เจตนา (Intent)", res["step_5_8"])
            with col_m2:
                extracted_info = f"{res['product']} | {res['attr']}" if res['product'] != "-" or res['attr'] != "-" else "ไม่พบข้อมูลเฉพาะ"
                st.metric("การสกัดข้อมูล", extracted_info)
            with col_m3:
                suffix_status = "ปรับตามหางเสียง" if res["has_suffix"] else "ไม่ใช้หางเสียง"
                st.metric("รูปแบบประโยค", suffix_status)

            with st.expander("ดูรายละเอียด NLP Pipeline 8 ขั้นตอน (Technical Details)"):
                st.write(f"**1. การทำความสะอาด (Clean):** `{res['step_5_1']}`")
                st.write(f"**2. การลดตัวอักษรซ้ำ (Reduced):** `{res['step_5_2']}`")
                st.write(f"**3. การปรับคำแสลง (Slang Normalization):** `{res['step_5_3']}`")
                st.write(f"**4. การตัดคำ (Word Tokenization):** `{res['step_5_4']}`")
                st.write(f"**5. การลบคำที่ไม่จำเป็น (Token Cleaning):** `{res['step_5_5']}`")
                st.write(f"**6. การปรับโครงสร้างประโยค (Sentence Adjustment):** `{res['step_5_6']}`")
                st.write(f"**7. การเพิ่มคำสุภาพ (Add Polite Particle):** `{res['step_5_7']}`")
                st.write(f"**8. การจำแนกเจตนา (Intent Classification):** `{res['step_5_8']}`")

        st.session_state.messages.append({"role": "bot", "content": res})

if __name__ == "__main__":
    main()
