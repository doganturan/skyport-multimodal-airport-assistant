from pathlib import Path
import os
import tempfile

import streamlit as st
from PIL import Image

from multimodal_assistant import MultimodalAssistant

PROJECT_DIR = Path(__file__).resolve().parent
KB_PATH = PROJECT_DIR / "data" / "airport_kb.json"
DEMO_DIR = PROJECT_DIR / "demo_assets"

DEMO_CASES = {
    "Text — Gate B12": {"text": "Where is Gate B12?"},
    "Voice — Train station": {"audio": "audio_07_transport.wav"},
    "Image — Special assistance": {"image": "special_assistance_05.png"},
    "Uncertainty — Unsupported greeting": {"text": "Hello there"},
}

st.set_page_config(page_title="SkyPort Airport Assistant", page_icon="✈️", layout="wide")

@st.cache_resource
def load_assistant():
    return MultimodalAssistant(kb_path=KB_PATH)

def run_request(text=None, image=None, audio_path=None):
    with st.spinner("Loading models and analysing the request..."):
        return load_assistant().process(text=text, image=image, audio_path=audio_path)

def show_result(result):
    st.subheader("Assistant Response")
    if result["status"] == "uncertain":
        st.error("No reliable airport service match could be identified.")
    elif result["status"] == "multimodal_mismatch":
        st.warning("The supplied inputs do not fully agree. The more specific text or voice request was used.")
    elif result["status"] == "category_only":
        st.info("The image identifies a service category, but not one unique airport location.")
    else:
        st.success("Airport service match found.")

    st.write(result["response"])

    if result["transcript"]:
        st.markdown("**Voice transcript**")
        st.write(result["transcript"])

    if result["record"]:
        record = result["record"]
        st.markdown("### Retrieved Service")
        st.write(f"**{record['name']}**")
        st.write(f"Terminal: {record['terminal']}")
        st.write(f"Location: {record['floor_zone']}")

    c1, c2 = st.columns(2)
    with c1:
        if result["text_similarity"] is not None:
            st.metric("Text similarity", f"{result['text_similarity']:.3f}")
    with c2:
        if result["image_similarity"] is not None:
            st.metric("Image similarity", f"{result['image_similarity']:.3f}")

    if result["text_intent"]:
        st.caption(f"Detected intent: {result['text_intent']}")
    if result["image_category"]:
        st.caption(f"Detected image category: {result['image_category']}")

st.title("✈️ SkyPort Airport Assistant")
st.caption("Multimodal passenger assistance using text, airport-sign images and voice.")

with st.sidebar:
    st.subheader("About")
    st.write("Academic proof-of-concept using a fictional airport knowledge base.")
    st.info("No live flight, gate or operational airport data is used.")

    st.subheader("Quick Demo")
    demo_choice = st.selectbox("Prepared example", list(DEMO_CASES))
    run_demo = st.button("Run Demo", use_container_width=True)
    st.caption("The first request can take longer while the ML models load.")

input_col, result_col = st.columns(2)

with input_col:
    st.subheader("Passenger Input")
    text_query = st.text_input("Ask a question", placeholder="Example: Where is Gate B12?")
    image_file = st.file_uploader("Upload an airport image", type=["png", "jpg", "jpeg"])
    recorded_audio = st.audio_input("Record a voice query")
    uploaded_audio = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a"])

    if image_file:
        st.image(image_file, caption="Uploaded image", use_container_width=True)

    audio_source = recorded_audio or uploaded_audio
    if audio_source:
        st.audio(audio_source)

    analyse = st.button("Ask Assistant", type="primary", use_container_width=True)

result = None

if run_demo:
    demo = DEMO_CASES[demo_choice]
    demo_text = demo.get("text")
    demo_image = Image.open(DEMO_DIR / demo["image"]).convert("RGB") if demo.get("image") else None
    demo_audio_path = str(DEMO_DIR / demo["audio"]) if demo.get("audio") else None
    result = run_request(text=demo_text, image=demo_image, audio_path=demo_audio_path)

    with input_col:
        st.caption(f"Quick demo: {demo_choice}")
        if demo_text:
            st.write(f"Text: {demo_text}")
        if demo.get("image"):
            st.image(DEMO_DIR / demo["image"], caption=demo["image"], use_container_width=True)
        if demo.get("audio"):
            st.audio(DEMO_DIR / demo["audio"])

elif analyse:
    if not text_query and not image_file and not audio_source:
        with input_col:
            st.warning("Please provide text, an image or a voice query.")
    else:
        image = Image.open(image_file).convert("RGB") if image_file else None
        audio_path = None
        if audio_source:
            suffix = ".wav" if recorded_audio else Path(audio_source.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as temp_audio:
                temp_audio.write(audio_source.getvalue())
                audio_path = temp_audio.name

        try:
            result = run_request(text=text_query or None, image=image, audio_path=audio_path)
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

if result is not None:
    with result_col:
        show_result(result)

st.divider()
st.caption("SkyPort International Airport — fictional academic proof-of-concept.")
