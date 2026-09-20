
import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INTENT_KEYWORDS = {
    "gate": ["gate", "departure gate"],
    "baggage_claim": ["baggage", "luggage", "suitcase", "bag collection", "claim my bag"],
    "check_in": ["check in", "check-in", "checkin", "check in desk", "check-in desk", "check in counter"],
    "security": ["security", "checkpoint", "screening"],
    "lounge": ["lounge"],
    "restaurant": ["restaurant", "food", "eat", "cafe", "bistro", "dining"],
    "transport": ["train", "railway", "taxi", "transport", "station"],
    "information": ["information", "info desk", "help desk", "ask for help"],
    "lost_and_found": ["lost", "lost property", "lost item", "left my"],
    "special_assistance": [
        "wheelchair", "special assistance", "passenger assistance",
        "reduced mobility", "mobility support"
    ],
}

INTENT_PRIORITY = [
    "special_assistance", "lost_and_found", "restaurant", "lounge",
    "baggage_claim", "check_in", "security", "transport", "information", "gate"
]

KNOWN_SERVICES = [
    "sky lounge", "business lounge", "airport bistro", "food court",
    "train station", "taxi pickup", "information desk", "special assistance"
]


class TextProcessor:
    def __init__(
        self,
        kb_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        self.kb_path = Path(kb_path)
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        self.model = SentenceTransformer(model_name, device=device)
        self.kb_texts = [self.record_to_text(record) for record in self.knowledge_base]
        self.kb_embeddings = self.model.encode(
            self.kb_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(self.kb_embeddings.shape[1])
        self.index.add(self.kb_embeddings)

    @staticmethod
    def preprocess_query(text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def record_to_text(record):
        related = ", ".join(record.get("related_facilities", []))
        category = record["category"].replace("_", " ")
        return (
            f"{record['name']}. Category: {category}. Terminal: {record['terminal']}. "
            f"Location: {record['floor_zone']}. {record['description']} "
            f"Directions: {record['directions']} Related facilities: {related}."
        )

    def detect_intent(self, text):
        cleaned = self.preprocess_query(text)
        matched = [
            intent
            for intent, keywords in INTENT_KEYWORDS.items()
            if any(keyword in cleaned for keyword in keywords)
        ]
        if not matched:
            return None
        return next((intent for intent in INTENT_PRIORITY if intent in matched), matched[0])

    def extract_entities(self, text):
        cleaned = self.preprocess_query(text)
        entities = {"gate": None, "terminal": None, "service": None}

        gate_match = re.search(r"\b(?:gate\s*)?([abc])[\s\-]?(\d{1,2})\b", cleaned)
        if gate_match:
            letter = gate_match.group(1).upper()
            number = int(gate_match.group(2))
            entities["gate"] = f"{letter}{number:02d}"

        terminal_match = re.search(r"\bterminal\s*([12])\b", cleaned)
        if terminal_match:
            entities["terminal"] = f"Terminal {terminal_match.group(1)}"

        entities["service"] = next(
            (service for service in KNOWN_SERVICES if service in cleaned),
            None,
        )
        return entities

    def retrieve(self, query, top_k=3):
        cleaned = self.preprocess_query(query)
        intent = self.detect_intent(query)
        entities = self.extract_entities(query)

        query_embedding = self.model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, len(self.knowledge_base))
        ranked = [
            {"record": self.knowledge_base[index], "similarity": float(score)}
            for score, index in zip(scores[0], indices[0])
        ]
        filtered = ranked

        if intent == "gate" and entities["gate"]:
            matches = [
                item for item in filtered
                if entities["gate"].lower() in item["record"]["name"].lower()
            ]
            if matches:
                filtered = matches
        else:
            if intent:
                matches = [item for item in filtered if item["record"]["category"] == intent]
                if matches:
                    filtered = matches

            if entities["gate"]:
                matches = [
                    item for item in filtered
                    if any(
                        entities["gate"].lower() in facility.lower()
                        for facility in item["record"].get("related_facilities", [])
                    )
                ]
                if matches:
                    filtered = matches

            if entities["terminal"]:
                matches = [
                    item for item in filtered
                    if item["record"]["terminal"] == entities["terminal"]
                ]
                if matches:
                    filtered = matches

            if entities["service"]:
                matches = [
                    item for item in filtered
                    if entities["service"] in item["record"]["name"].lower()
                ]
                if matches:
                    filtered = matches

        results = []
        for item in filtered[:top_k]:
            record = item["record"]
            results.append({
                "record_id": record["record_id"],
                "name": record["name"],
                "category": record["category"],
                "terminal": record["terminal"],
                "floor_zone": record["floor_zone"],
                "description": record["description"],
                "directions": record["directions"],
                "opening_hours": record["opening_hours"],
                "accessibility": record["accessibility"],
                "similarity": item["similarity"],
            })

        return {
            "query": query,
            "preprocessed_query": cleaned,
            "intent": intent,
            "entities": entities,
            "results": results,
        }
