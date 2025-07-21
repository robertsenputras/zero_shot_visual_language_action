from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    model_name: str = "models/owlv2-base-patch16"
    device: str = "cpu"
    detection_threshold: float = 0.22
    nms_threshold: float = 0.3
    nms_sigma: float = 0.5
    visualization_score_threshold: float = 0.22

# Define queries for different types of objects to detect
PILE_QUERIES: List[str] = [
    # generic "pile" to catch anything
    "pile", 

    # sand/fines
    "pile of sand", "sand heap",

    # gravel / fine aggregate
    "gravel pile", "heap of gravel", "gravel stockpile", "aggregate pile",

    # crushed stone / rock
    "crushed rock pile", "stone pile", "rock heap",

    # generic earth
    "earth mound", "soil mound",

    # (optional) catch-all "spoil" jargon
    "spoil pile", "spoil heap",

    # Loader bucket
    "loader bucket", "excavator bucket",
] 