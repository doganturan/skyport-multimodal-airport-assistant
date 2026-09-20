import gc

from audio_processor import AudioProcessor
from image_processor import ImageProcessor
from text_processor import TextProcessor


class MultimodalAssistant:
    def __init__(self, kb_path, device=None):
        self.kb_path = kb_path
        self.device = device
        self.text_processor = TextProcessor(kb_path, device=device)

    @staticmethod
    def build_response(record):
        return (
            f"{record['name']}. {record['description']} "
            f"Directions: {record['directions']} "
            f"Opening hours: {record['opening_hours']} "
            f"Accessibility: {record['accessibility']}"
        )

    def process(self, text=None, image=None, audio_path=None):
        transcript = None
        image_result = None

        if audio_path:
            audio_processor = AudioProcessor(
                self.kb_path,
                device=self.device,
                text_processor=self.text_processor,
            )
            transcript = audio_processor.transcribe(audio_path)
            del audio_processor
            gc.collect()

        query = (text or transcript or "").strip()
        text_result = self.text_processor.retrieve(query, top_k=3) if query else None

        if image is not None:
            image_processor = ImageProcessor(self.kb_path, device=self.device)
            image_result = image_processor.analyse(image, top_k=3)
            del image_processor
            gc.collect()

        result = {
            "input_text": text,
            "transcript": transcript,
            "text_intent": text_result["intent"] if text_result else None,
            "image_category": image_result["best_category"] if image_result else None,
            "text_similarity": None,
            "image_similarity": None,
            "status": "uncertain",
            "record": None,
            "response": "No reliable airport service match.",
        }

        if text_result and text_result["results"]:
            result["text_similarity"] = text_result["results"][0]["similarity"]
        if image_result:
            result["image_similarity"] = image_result["predictions"][0]["similarity"]

        if text_result and text_result["intent"] and text_result["results"]:
            record = text_result["results"][0]
            result["status"] = "matched"
            if image_result:
                result["status"] = (
                    "multimodal_consistent"
                    if record["category"] == image_result["best_category"]
                    else "multimodal_mismatch"
                )
            result["record"] = record
            result["response"] = self.build_response(record)
            return result

        if image_result:
            candidates = image_result["kb_candidates"]
            if len(candidates) == 1:
                record = candidates[0]
                result.update({
                    "status": "image_match",
                    "record": record,
                    "response": self.build_response(record),
                })
            else:
                names = ", ".join(record["name"] for record in candidates)
                category = image_result["best_category"].replace("_", " ")
                result["status"] = "category_only"
                result["response"] = (
                    f"The image appears to show {category}. "
                    f"Possible matching services: {names}."
                )

        return result
