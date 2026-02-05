import streamlit as st
import cv2
from ultralytics import YOLO
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🔥 Fire Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.status-safe {
    background-color: #0f5132;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    text-align: center;
}
.status-danger {
    background-color: #842029;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    text-align: center;
}
.metric-box {
    background-color: #1f2933;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Control Panel")

run = st.sidebar.toggle("▶️ Start Camera", value=False)
show_bbox = st.sidebar.checkbox("📦 Show Bounding Box", True)
enable_alert = st.sidebar.checkbox("🚨 Enable Alert", True)

st.sidebar.markdown("---")
st.sidebar.info("🔥 Fire Detection System\nYOLOv8")

# ===============================
# MAIN TITLE
# ===============================
st.title("🔥 Fire Detection Live Dashboard")

# ===============================
# METRICS ROW
# ===============================
col1, col2, col3 = st.columns(3)

fps_box = col1.empty()
det_box = col2.empty()
status_box = col3.empty()

# ===============================
# VIDEO PLACEHOLDER
# ===============================
frame_window = st.empty()

# ===============================
# LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

# ===============================
# CAMERA SETUP
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    st.error("❌ Kamera tidak dapat dibuka!")

alert_cooldown = 0

# ===============================
# MAIN LOOP - WHILE (TIDAK PAKAI st.rerun)
# ===============================
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("❌ Frame tidak terbaca")
        break

    start = time.time()

    # YOLO Detection
    results = model(frame, conf=0.5, imgsz=416, verbose=False)

    if show_bbox:
        frame = results[0].plot(labels=False)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame, channels="RGB", use_container_width=True)

    # Detection Count
    detected = len(results[0].boxes) if results[0].boxes else 0

    # FPS
    elapsed = time.time() - start
    fps = int(1 / elapsed) if elapsed > 0 else 0

    # ===============================
    # UPDATE METRICS
    # ===============================
    fps_box.markdown(
        f"<div class='metric-box'>⚡ FPS<br><h2>{fps}</h2></div>",
        unsafe_allow_html=True
    )

    det_box.markdown(
        f"<div class='metric-box'>🔥 Detected Objects<br><h2>{detected}</h2></div>",
        unsafe_allow_html=True
    )

    # STATUS
    if detected > 0:
        status_box.markdown(
            "<div class='status-danger'>🔥 FIRE DETECTED</div>",
            unsafe_allow_html=True
        )

        if enable_alert and time.time() - alert_cooldown > 5:
            st.toast("🔥 PERINGATAN KEBAKARAN!", icon="🚨")
            alert_cooldown = time.time()
    else:
        status_box.markdown(
            "<div class='status-safe'>✅ SAFE</div>",
            unsafe_allow_html=True
        )

    time.sleep(0.01)
    # TIDAK ADA st.rerun() di sini!

# ===============================
# STOP CAMERA
# ===============================
if not run:
    cap.release()
    cv2.destroyAllWindows()
    
    frame_window.empty()
    fps_box.markdown(
        "<div class='metric-box'>⚡ FPS<br><h2>0</h2></div>",
        unsafe_allow_html=True
    )
    det_box.markdown(
        "<div class='metric-box'>🔥 Detected Objects<br><h2>0</h2></div>",
        unsafe_allow_html=True
    )
    status_box.markdown(
        "<div class='metric-box'>🛑 Camera Stopped</div>",
        unsafe_allow_html=True
    )