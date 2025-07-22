# inference_qwen_planner.py
# Implements end-to-end inference pipeline combining OWL-VLM object detection with
# fine-tuned Qwen2.5 action planner for wheel loader operations.

import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from model import ObjectDetector
from config import ModelConfig, PILE_QUERIES
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model with quantization config
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=quantization_config,
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

def plan_action(model, tokenizer, command: str, detections: list, max_new_tokens: int = 256) -> dict:
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

    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                temperature=0.1
            )

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract everything after "Output:" and before any potential system message
        json_str = output_text.split("Output:")[-1].split("System:")[0].strip()
        
        # Try to find the complete JSON object
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx + 1]
        
        try:
            plan = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {str(e)}")
            print(f"Raw output: {json_str}")
            
            # Attempt to fix common JSON issues
            if json_str.endswith('",') or json_str.endswith('"}'):
                json_str = json_str.rstrip(',"')
                if not json_str.endswith("}"):
                    json_str += "}"
            
            # Try parsing again
            plan = json.loads(json_str)
        
        # Validate plan structure
        if "plan" not in plan:
            plan = {"plan": plan}
        
        return plan
        
    except Exception as e:
        print(f"Error generating plan: {str(e)}")
        print(f"Falling back to default plan structure...")
        
        # Generate a basic plan using the first detection
        if detections:
            return {
                "plan": [
                    {
                        "step": 1,
                        "action": "drive",
                        "target": detections[0]["id"],
                        "coords": detections[0]["coords"]
                    },
                    {
                        "step": 2,
                        "action": "dig",
                        "target": detections[0]["id"],
                        "coords": detections[0]["coords"]
                    },
                    {
                        "step": 3,
                        "action": "stop"
                    }
                ]
            }
        else:
            return {"plan": [{"step": 1, "action": "stop"}]}

def visualize_results(image_path: str, detection_result, processed_detections, plan=None, save_path=None):
    """
    Enhanced visualization of detection results and planned actions.
    
    Args:
        image_path: Path to input image
        detection_result: Raw detection results from OWL-VLM
        processed_detections: Processed and normalized detections
        plan: Optional action plan to visualize
        save_path: Path to save the visualization
    """
    # Load and convert image if needed
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Set up the figure with a white background
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 10), facecolor='white')
    
    # Create subplots with proper spacing
    if plan:
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
    else:
        ax1 = plt.subplot(111)
    
    # Show original image with detections
    ax1.imshow(img)
    ax1.set_title("Detected Piles", fontsize=14, pad=20)
    
    # Draw detections with improved visibility
    height, width = img.shape[:2]
    for det in processed_detections:
        # Convert normalized coordinates back to pixel space
        x = int(det["coords"][0] * width)
        y = int(det["coords"][1] * height)
        
        # Draw detection point with larger marker and outline
        ax1.plot(x, y, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2, label=det["id"])
        
        # Add text with better visibility
        text_box = dict(
            facecolor='black',
            alpha=0.8,
            edgecolor='white',
            boxstyle='round,pad=0.5'
        )
        ax1.text(
            x + width*0.02,  # Offset by 2% of image width
            y + height*0.02,  # Offset by 2% of image height
            f'{det["id"]}\nconf: {det["confidence"]:.2f}',
            color='white',
            fontsize=12,
            bbox=text_box
        )
    
    # Add legend with better visibility
    ax1.legend(
        bbox_to_anchor=(0.5, -0.05),
        loc='upper center',
        ncol=len(processed_detections),
        fontsize=12,
        facecolor='white',
        edgecolor='black'
    )
    
    # If we have a plan, visualize it
    if plan and "plan" in plan:
        ax2.imshow(img)
        ax2.set_title("Action Plan", fontsize=14, pad=20)
        
        # Draw planned actions
        for step in plan["plan"]:
            if "coords" in step:
                x = int(step["coords"][0] * width)
                y = int(step["coords"][1] * height)
                
                # Different colors for different actions with better visibility
                color_map = {
                    "drive": ('g', 'o', 'Drive'),
                    "dig": ('r', 's', 'Dig'),
                    "stop": ('b', '^', 'Stop')
                }
                color, marker, action_name = color_map.get(
                    step["action"],
                    ('k', 'x', step["action"])  # default
                )
                
                # Draw action point with larger marker and outline
                ax2.plot(
                    x, y,
                    color + marker,
                    markersize=15,
                    markeredgecolor='white',
                    markeredgewidth=2,
                    label=f'Step {step["step"]}: {action_name}'
                )
                
                # Add text with better visibility
                text_box = dict(
                    facecolor='black',
                    alpha=0.8,
                    edgecolor='white',
                    boxstyle='round,pad=0.5'
                )
                ax2.text(
                    x + width*0.02,
                    y + height*0.02,
                    f'Step {step["step"]}\n{action_name}',
                    color='white',
                    fontsize=12,
                    bbox=text_box
                )
        
        # Add legend with better visibility
        ax2.legend(
            bbox_to_anchor=(0.5, -0.05),
            loc='upper center',
            ncol=len([s for s in plan["plan"] if "coords" in s]),
            fontsize=12,
            facecolor='white',
            edgecolor='black'
        )
    
    # Remove axes for cleaner look
    for ax in fig.get_axes():
        ax.axis('off')
    
    # Adjust layout and add padding
    plt.tight_layout(pad=3.0)
    
    # Save figure if path provided
    if save_path:
        # Ensure the figure is rendered with a white background
        fig.patch.set_facecolor('white')
        
        # Save with high quality settings
        plt.savefig(
            save_path,
            bbox_inches='tight',
            dpi=300,
            facecolor=fig.get_facecolor(),
            edgecolor='none'
        )
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    plt.close()

def main():
    """Main execution function for the inference pipeline."""
    parser = argparse.ArgumentParser(description="Wheel loader action planner with visual detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--command", required=True, help="Natural language command for the loader")
    parser.add_argument("--visualize", action="store_true", help="Visualize detections")
    parser.add_argument("--output", help="Path to save the output JSON plan")
    parser.add_argument("--save_viz", help="Path to save visualization (e.g., 'output.png', 'output.jpg')")
    args = parser.parse_args()

    # Initialize vision and language models
    planner_model, tokenizer = load_planner_model()
    detector = ObjectDetector(ModelConfig())

    # Perform pile detection
    print("Running object detection...")
    detection_result = detector.detect(args.image, PILE_QUERIES)
    
    # Process detections for action planning
    processed_detections = process_detections(detection_result, detector.load_image(args.image).shape)
    
    if not processed_detections:
        print("Warning: No piles detected in the image!")
        return
    
    # Generate action sequence
    print("Generating action plan...")
    plan = plan_action(planner_model, tokenizer, args.command, processed_detections)
    
    # Visualize results
    if args.visualize or args.save_viz:
        try:
            visualize_results(
                args.image,
                detection_result,
                processed_detections,
                plan,
                args.save_viz
            )
        except Exception as e:
            print(f"Warning: Visualization failed: {str(e)}")
    
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
