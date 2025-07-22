# Zero-Shot Visual Language Action Planner for Wheel Loader Operations

This repository implements an intelligent action planner for wheel loader operations using a combination of Vision-Language Models (VLM) and Large Language Models (LLM). The system can interpret natural language commands and visual input to generate appropriate action sequences for wheel loader operations.

## Project Overview

The project aims to enhance a small LLM (Qwen-2.5-0.5B) to predict actions for wheel loader operations by:
1. Understanding visual scenes using zero-shot VLM capabilities
2. Processing natural language commands
3. Generating appropriate action sequences

### Key Features

- Zero-shot visual understanding using OWL-VLM
- Natural language command processing
- Action sequence generation for wheel loader operations
- QLoRA fine-tuning for efficient model adaptation

## Implementation Details

### 1. Vision-Language Processing
- **Model**: OWL-VLM for zero-shot visual understanding
- **Capabilities**: 
  - Pile detection
  - Scene understanding
  - Position extraction

### 2. Dataset Creation
- **Source**: YouTube construction site footage
- **Processing Pipeline**:
  1. Video scraping from relevant construction footage
  2. Frame extraction
  3. VLM-based scene analysis
  4. Automated input-output pair generation

### 3. Model Architecture
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Fine-tuning**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for efficient training
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 16
  - Target modules: q_proj, v_proj
  - Dropout: 0.05

## Training Setup

The model is trained using the following configuration:

```python
- Max sequence length: 384
- Batch size: 16
- Learning rate: 1e-4
- Training epochs: 1
- Warmup ratio: 0.03
- Learning rate scheduler: Cosine
```

### Input Format

The model expects inputs in the following format:
```json
User: {
  "command": "Fill the shovel.",
  "detections": [
    {"id":"pile1","coords":[0.4,0.7]},
    {"id":"pile2","coords":[0.8,0.3]}
  ]
}

```

### Output Format

The model generates action plans in JSON format:
```json
{
  "plan": [
    {"step":1,"action":"drive","target":"pile1","coords":[0.4,0.7]},
    {"step":2,"action":"dig",  "target":"pile1","coords":[0.4,0.7]},
    {"step":3,"action":"stop"}
  ]
}

```

## Usage

1. **Prerequisites**
```bash
# Install Git LFS
## Ubuntu/Debian
sudo apt-get install git-lfs

## macOS (using Homebrew)
brew install git-lfs

## Windows (using Chocolatey)
choco install git-lfs
```

2. **Installation**
```bash
# Clone the repository
git clone <repository_url>
cd zero_shot_visual_language_action

# Setup Git LFS and pull files
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

3. **Training**
```bash
python train_qlora.py
```

4. **Running Inference**
The inference script (`inference_qwen_planner.py`) combines OWL-VLM detection with the trained planner model to generate action sequences. Here are the available options:

```bash
python inference_qwen_planner.py --image <path_to_image> --command <command> [options]

Required arguments:
  --image PATH          Path to input image
  --command TEXT        Natural language command for the loader (e.g., "Fill the shovel")

Optional arguments:
  --visualize          Show visualization of detected piles
  --output PATH        Save the generated plan to a JSON file
```

Example usage:
```bash
# Basic usage - prints plan to console
python inference_qwen_planner.py --image examples/site1.jpg --command "Fill the shovel"

# With visualization
python inference_qwen_planner.py --image examples/site1.jpg --command "Fill the shovel" --visualize

# Save output to file
python inference_qwen_planner.py --image examples/site1.jpg --command "Fill the shovel" --output plan.json
```

The output will be a JSON plan with the following format:
```json
{
  "plan": [
    {"step":1,"action":"drive","target":"pile1","coords":[0.4,0.7]},
    {"step":2,"action":"dig",  "target":"pile1","coords":[0.4,0.7]},
    {"step":3,"action":"stop"}
  ]
}
```

Where:
- `coords`: Normalized coordinates [x, y] in range [0,1]
- `target`: ID of the detected pile
- `action`: One of ["drive", "dig", "stop"]
- `step`: Sequential step number

## Project Pipeline

1. **Visual Processing**
   - Input: Construction site images
   - Process: Zero-shot VLM analysis
   - Output: Scene understanding and object positions

2. **Command Processing**
   - Input: Natural language commands
   - Process: Language understanding and context mapping
   - Output: Structured command interpretation

3. **Action Planning**
   - Input: Visual analysis + Command interpretation
   - Process: Fine-tuned LLM processing
   - Output: Actionable sequence of operations

## Future Improvements

1. **Model Enhancement**
   - Expand training dataset
   - Implement multi-task learning
   - Add more action types and scenarios

2. **System Integration**
   - Real-time processing pipeline
   - Robot control interface
   - Safety verification layer

3. **Evaluation Framework**
   - Metrics for action accuracy
   - Performance benchmarking
   - Safety compliance checking

## Notes

- The model includes a graceful shutdown mechanism (Ctrl+C handler) to save progress during training
- Current implementation focuses on basic operations (locating piles, driving, digging)
- System is designed for research and prototype purposes
