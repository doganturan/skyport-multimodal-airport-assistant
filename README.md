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

## What can the assistant help with?

The prototype can answer questions about the main services included in the SkyPort knowledge base:

- Gates
- Baggage claim
- Check-in areas
- Security
- Airport lounges
- Restaurants and food areas
- Train and taxi transport
- Information desks
- Lost and Found
- Special assistance

Example questions:

- `Where is Gate B12?`
- `Where is baggage claim in Terminal 2?`
- `Where can I check in at Terminal 1?`
- `How do I reach security in Terminal 2?`
- `Is there a lounge in Terminal 2?`
- `Where can I get food in Terminal 1?`
- `How do I get to the train station?`
- `Where can I get a taxi?`
- `Where is the information desk?`
- `Where can I report a lost bag?`
- `I need wheelchair assistance.`

The same types of questions can also be spoken as voice input. Airport-sign images can be uploaded to identify a matching service category, and text can be combined with an image to give extra context.

## Live demo

https://skyport-doganturan.streamlit.app/

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
