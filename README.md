# SkyPort Multimodal Airport Assistant

A multimodal airport passenger-assistance prototype for the MSc Artificial Intelligence Multi-Modal Chatbots module.

## Features

- Text questions
- Airport-sign image upload
- Voice recording or audio upload
- Combined multimodal input
- MiniLM + FAISS text retrieval
- Whisper speech-to-text
- CLIP + FAISS image-category matching
- Decision-level multimodal fusion
- Fictional 20-record airport knowledge base

## Live demo

The Streamlit Community Cloud URL will be added here after deployment.

The sidebar includes prepared text, voice and image demo scenarios. The first request may take longer while the models are loaded.

Suggested manual tests:
- Text: `Where is Gate B12?`
- Voice: say `Where is the airport train station?`
- Image: use the prepared Special Assistance demo
- Uncertainty: `Hello there`

## Run locally

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

FFmpeg is required for audio support.

## Note

SkyPort International Airport is fictional. This app does not provide live flight, gate, safety or operational airport information.
