
import librosa
import torch
import whisper

from text_processor import TextProcessor


class AudioProcessor:
    def __init__(
        self,
        kb_path,
        model_size="base",
        device=None,
        text_processor=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        self.text_processor = text_processor or TextProcessor(kb_path, device=self.device)

    def transcribe(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        result = self.whisper_model.transcribe(
            audio,
            language="en",
            fp16=(self.device == "cuda"),
        )
        return result["text"].strip()

    def analyse(self, audio_path):
        transcript = self.transcribe(audio_path)
        retrieval = self.text_processor.retrieve(transcript, top_k=1)

        if retrieval["intent"] is None:
            return {
                "transcript": transcript,
                "intent": None,
                "entities": retrieval["entities"],
                "results": [],
                "status": "uncertain",
            }

        return {
            "transcript": transcript,
            "intent": retrieval["intent"],
            "entities": retrieval["entities"],
            "results": retrieval["results"],
            "status": "matched",
        }
