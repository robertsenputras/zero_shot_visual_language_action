# generate_dataset.py
# Processes extracted frames using OWL-VLM to create training data for the action planner.
# Generates input-output pairs by combining visual detections with templated commands.

import os
import json
import random
import argparse
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import ObjectDetector
from config import ModelConfig, PILE_QUERIES

class DatasetGenerator:
    """
    Generates training data by combining OWL-VLM detections with action templates.
    Processes frames to create input-output pairs for action planner training.
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initializes dataset generator with detection pipeline.
        
        Args:
            config: Configuration for the OWL-VLM detector
        """
        if config is None:
            config = ModelConfig()
        self.detector = ObjectDetector(config)
        
        # Command templates for data generation
        self.command_templates = [
            "Fill the shovel.",
            "Load material from the nearest pile.",
            "Go to the closest pile and dig.",
            "Pick up material from pile {pile_id}.",
            "Drive to {pile_id} and fill the bucket.",
            "Get material from the pile at {coords}."
        ]

    def process_frame(self, image_path: str) -> Dict[str, Any]:
        """
        Processes a single frame to extract pile detections.
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict: Processed detections with normalized coordinates
        """
        # Run detection
        result = self.detector.detect(image_path, PILE_QUERIES)
        
        # Process and normalize detections
        processed = []
        img = self.detector.load_image(image_path)
        height, width = img.shape[:2]
        
        for i, (box, score, label) in enumerate(zip(
            result.boxes, result.scores, result.labels
        )):
            if score < 0.3:  # Confidence threshold
                continue
                
            # Skip unreasonably large detections
            if (box[2]-box[0]) > 0.75*width or (box[3]-box[1]) > 0.75*height:
                continue
            
            # Calculate normalized center coordinates
            x_center = ((box[0] + box[2]) / 2) / width
            y_center = ((box[1] + box[3]) / 2) / height
            
            processed.append({
                "id": f"pile{i+1}",
                "coords": [float(x_center), float(y_center)],
                "confidence": float(score),
                "type": PILE_QUERIES[label]
            })
        
        return processed

    def generate_action_sequence(
        self,
        detections: List[Dict[str, Any]],
        target_pile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates a sequence of actions to interact with target pile.
        
        Args:
            detections: List of detected piles
            target_pile: Selected pile for interaction
            
        Returns:
            list: Sequence of action steps
        """
        return [
            {
                "step": 1,
                "action": "drive",
                "target": target_pile["id"],
                "coords": target_pile["coords"]
            },
            {
                "step": 2,
                "action": "dig",
                "target": target_pile["id"],
                "coords": target_pile["coords"]
            },
            {
                "step": 3,
                "action": "stop"
            }
        ]

    def generate_example(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates a single training example with input and output.
        
        Args:
            detections: List of detected piles
            
        Returns:
            dict: Training example with command and action sequence
        """
        if not detections:
            return None
            
        # Select target pile and template
        target_pile = random.choice(detections)
        template = random.choice(self.command_templates)
        
        # Format command
        command = template.format(
            pile_id=target_pile["id"],
            coords=f"[{target_pile['coords'][0]:.2f}, {target_pile['coords'][1]:.2f}]"
        )
        
        # Generate example
        return {
            "input": {
                "command": command,
                "detections": detections
            },
            "output": {
                "plan": self.generate_action_sequence(detections, target_pile)
            }
        }

    def process_dataset(
        self,
        frames_dir: str,
        output_file: str = "final_trunc.jsonl",
        max_examples: int = None
    ):
        """
        Processes all frames to generate the complete training dataset.
        
        Args:
            frames_dir: Directory containing extracted frames
            output_file: Path to save generated dataset
            max_examples: Maximum number of examples to generate
        """
        frames = list(Path(frames_dir).glob("**/*.jpg"))
        if max_examples:
            frames = frames[:max_examples]
            
        print(f"Generating dataset from {len(frames)} frames...")
        
        with open(output_file, 'w') as f:
            for frame_path in tqdm(frames):
                # Process frame
                detections = self.process_frame(str(frame_path))
                if not detections:
                    continue
                    
                # Generate multiple examples per frame
                for _ in range(min(len(detections), 3)):  # Up to 3 examples per frame
                    example = self.generate_example(detections)
                    if example:
                        f.write(json.dumps(example) + '\n')

def main():
    """Main execution function for dataset generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate training dataset from frames")
    parser.add_argument(
        "--frames",
        required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--output",
        default="final_trunc.jsonl",
        help="Output JSONL file for training data"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="Maximum number of examples to generate"
    )
    args = parser.parse_args()
    
    # Initialize and run generator
    generator = DatasetGenerator()
    generator.process_dataset(
        args.frames,
        args.output,
        args.max_examples
    )

if __name__ == "__main__":
    main()
