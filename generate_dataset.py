# generate_finetune_dataset_randomized.py
# Script to build a QLoRA fine-tuning JSONL dataset with randomized and paraphrased commands

import glob
import json
import random
import re
import argparse
from pathlib import Path
from tqdm import tqdm

class RoundingFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            return round(obj, 2)
        return super().default(obj)

# Optional paraphrasing pipeline
try:
    from transformers import pipeline
    _paraphraser = pipeline("text2text-generation", model="t5-base", device=0)
    PARAPHRASER_AVAILABLE = True
except Exception:
    PARAPHRASER_AVAILABLE = False

# --- Synonym pools for rule-based generation ---
INTROS = ["", "Please", "Kindly", "Could you"]
VERBS = ["drive", "move", "head over", "go"]
PREPS = [
    "to the nearest pile",
    "to the closest pile",
    "to the next pile",
    "towards the closest pile",
    "to the pile in front"
]
ACTIONS = [
    "fill the bucket",
    "scoop material",
    "dig material",
    "collect material",
    "load material"
]
GENERIC_COMMANDS = [
    "Fill the shovel.",
    "Dig out material from the next pile.",
    "Go to the nearest pile and dig.",
    "Move to the closest pile.",
    "Load material from the nearest pile."
]

# --- Helper functions ---

def select_pile_id(command: str, detections: list) -> str:
    """
    Choose pile ID based on command (always selects nearest/next pile).
    """
    def pile_num(d): return int(d['id'].replace('pile',''))
    dets = sorted(detections, key=pile_num)
    if not dets:
        raise ValueError("No piles detected")
    
    # Always select the first pile as "nearest/next"
    return dets[0]['id']


def build_plan(pile_id: str, detections: list) -> dict:
    """
    Construct a 3-step plan for the given pile_id.
    """
    det = next(d for d in detections if d['id'] == pile_id)
    coords = det.get('center', det.get('coords'))
    # Round coordinates to 2 decimal places
    if coords:
        coords = [round(float(c), 2) for c in coords]
    return {
        "plan": [
            {"step": 1, "action": "drive", "target": pile_id, "coords": coords},
            {"step": 2, "action": "dig",   "target": pile_id, "coords": coords},
            {"step": 3, "action": "stop"}
        ]
    }


def generate_rule_commands(n: int, max_base_commands: int = None) -> list:
    """
    Generate rule-based command variants.
    Args:
        n: pile number (unused now)
        max_base_commands: if set, randomly sample this many base commands
    """
    cmds = []
    # Always include generic commands
    cmds += GENERIC_COMMANDS
    
    # Generate specific commands
    specific_cmds = []
    for intro in INTROS:
        for verb in VERBS:
            for prep in PREPS:
                for act in ACTIONS:
                    # build sentence
                    intro_str = (intro + ' ') if intro else ''
                    cmd = f"{intro_str}{verb} {prep} and {act}."
                    # capitalize first letter
                    specific_cmds.append(cmd[0].upper() + cmd[1:])
    
    # Random sampling if max_base_commands is set
    if max_base_commands and len(specific_cmds) > max_base_commands - len(GENERIC_COMMANDS):
        specific_cmds = random.sample(specific_cmds, max_base_commands - len(GENERIC_COMMANDS))
    
    cmds.extend(specific_cmds)
    return cmds


def paraphrase_commands(cmds: list, num_paraphrases: int=1) -> list:
    """
    Use a paraphrasing model to expand each command.
    Returns original + paraphrases.
    """
    if not PARAPHRASER_AVAILABLE:
        return cmds
    new_cmds = []
    for cmd in cmds:
        new_cmds.append(cmd)
        # Configure generation parameters based on num_paraphrases
        gen_kwargs = {
            "max_length": 64,
            "num_beams": max(4, num_paraphrases * 2),  # Use enough beams
            "num_return_sequences": num_paraphrases,
            "temperature": 0.7,  # Add some randomness
            "do_sample": True    # Enable sampling
        }
        outputs = _paraphraser(cmd, **gen_kwargs)
        for out in outputs:
            text = out.get('generated_text', '').strip()
            if text and text.lower() != cmd.lower():
                new_cmds.append(text)
    return list(dict.fromkeys(new_cmds))  # dedupe, preserve order


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSONL dataset for QLoRA fine-tuning with randomized commands"
    )
    parser.add_argument('--detections_dir', default='detections',
                        help='Directory of per-frame detection JSONs')
    parser.add_argument('--output', default='finetune.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--paraphrase', action='store_true',
                        help='Enable paraphrasing via T5 model')
    parser.add_argument('--num-paraphrases', type=int, default=1,
                        help='Number of paraphrases per original command')
    parser.add_argument('--max-base-commands', type=int, default=20,
                        help='Maximum number of base commands to generate per pile (before paraphrasing)')
    args = parser.parse_args()

    det_files = sorted(Path(args.detections_dir).glob('*.json'))
    if not det_files:
        print(f"No detection files in {args.detections_dir}")
        return

    total = 0
    with open(args.output, 'w') as fout:
        for det_file in tqdm(det_files, desc="Frames"):
            try:
                # Load and process detections, rounding any floats
                with open(det_file) as f:
                    detections = json.load(f)
                for det in detections:
                    if 'center' in det:
                        det['center'] = [round(float(c), 2) for c in det['center']]
                    if 'coords' in det:
                        det['coords'] = [round(float(c), 2) for c in det['coords']]
                    if 'bbox' in det:
                        det['bbox'] = [round(float(c), 2) for c in det['bbox']]

                # skip if no valid detections
                if not detections:
                    continue
                # generate commands for each pile
                all_cmds = []
                for det in detections:
                    n = int(det['id'].replace('pile',''))
                    rule_cmds = generate_rule_commands(n, args.max_base_commands)
                    if args.paraphrase:
                        rule_cmds = paraphrase_commands(rule_cmds, args.num_paraphrases)
                    all_cmds.extend(rule_cmds)
                # dedupe
                unique_cmds = list(dict.fromkeys(all_cmds))
                # build records
                for cmd in unique_cmds:
                    try:
                        pile_id = select_pile_id(cmd, detections)
                        plan = build_plan(pile_id, detections)
                        if plan is None:
                            continue
                        record = {
                            "input": {"command": cmd, "detections": detections},
                            "output": plan
                        }
                        # Use custom encoder to round all floats
                        fout.write(json.dumps(record, cls=RoundingFloatEncoder) + "\n")
                        total += 1
                    except Exception:
                        continue
            except Exception as e:
                print(f"Warning: Error processing {det_file}: {str(e)}")
                continue

    print(f"Wrote {total} examples to {args.output}")

if __name__ == '__main__':
    main()
