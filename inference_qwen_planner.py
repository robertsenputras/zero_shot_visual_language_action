# inference_qwen_planner.py
# Implements end-to-end inference pipeline combining OWL-VLM object detection with
# fine-tuned Qwen2.5 action planner for wheel loader operations.

import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from model import ObjectDetector
from config import ModelConfig, PILE_QUERIES

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Base instruction-tuned model
LORA_WEIGHTS = "qlora-qwen2.5-planner"  # Path to fine-tuned LoRA weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_planner_model():
    """
    Initializes the action planner model with 4-bit quantization and LoRA adapters.
    Returns:
        tuple: (model, tokenizer) - The loaded model and its tokenizer
    """
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        quantization_config={
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.float16
        },
        trust_remote_code=True
    )

    print("Applying LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
    model.eval()
    
    return model, tokenizer

def process_detections(detection_result, image_shape):
    """
    Converts OWL-VLM detection results to planner-compatible format.
    
    Args:
        detection_result: Detection results from OWL-VLM
        image_shape: Original image dimensions (height, width)
    
    Returns:
        list: Processed detections with normalized coordinates and confidence scores
    """
    detections = []
    height, width = image_shape[:2]
    
    for i, (box, score, label) in enumerate(zip(
        detection_result.boxes,
        detection_result.scores,
        detection_result.labels
    )):
        # Filter low confidence detections
        if score < 0.3:
            continue
            
        # Convert box coordinates to normalized center points
        x_center = ((box[0] + box[2]) / 2) / width
        y_center = ((box[1] + box[3]) / 2) / height
        
        detections.append({
            "id": f"pile{i+1}",
            "type": PILE_QUERIES[label],
            "coords": [float(x_center), float(y_center)],
            "confidence": float(score)
        })
    
    return detections

def plan_action(model, tokenizer, command: str, detections: list, max_new_tokens: int = 128) -> dict:
    """
    Generates an action plan based on natural language command and scene detections.
    
    Args:
        model: Fine-tuned planner model
        tokenizer: Model tokenizer
        command: Natural language command
        detections: List of processed pile detections
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        dict: Generated action plan in JSON format
    """
    system_prompt = (
        "System: You are a wheel-loader action planner. "
        "Given a JSON of detections and a command, generate a JSON plan."
    )
    input_payload = {"command": command, "detections": detections}
    prompt = (
        f"{system_prompt}\n"
        f"Input: {json.dumps(input_payload, ensure_ascii=False)}\n"
        "Output:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    json_str = output_text.split("Output:")[-1].strip()
    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse plan JSON: {json_str}")
    return plan

def main():
    """Main execution function for the inference pipeline."""
    parser = argparse.ArgumentParser(description="Wheel loader action planner with visual detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--command", required=True, help="Natural language command for the loader")
    parser.add_argument("--visualize", action="store_true", help="Visualize detections")
    parser.add_argument("--output", help="Path to save the output JSON plan")
    args = parser.parse_args()

    # Initialize vision and language models
    planner_model, tokenizer = load_planner_model()
    detector = ObjectDetector(ModelConfig())

    # Perform pile detection
    print("Running object detection...")
    detection_result = detector.detect(args.image, PILE_QUERIES)
    
    # Visualize detections if requested
    if args.visualize:
        detector.visualize(args.image, detection_result)
    
    # Process detections for action planning
    processed_detections = process_detections(detection_result, detector.load_image(args.image).shape)
    
    if not processed_detections:
        print("Warning: No piles detected in the image!")
    
    # Generate action sequence
    print("Generating action plan...")
    plan = plan_action(planner_model, tokenizer, args.command, processed_detections)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"Plan saved to {args.output}")
    else:
        print("Generated plan:")
        print(json.dumps(plan, indent=2))

if __name__ == "__main__":
    main()
