# scrape_data_pipeline.py
# End-to-end script: download YouTube videos, sample frames, detect piles, save JSON per frame

import os
import subprocess
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import torch

from model import ObjectDetector
from config import ModelConfig, PILE_QUERIES

class PileDetectionPipeline:
    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(device="cuda" if torch.cuda.is_available() else "cpu")
        self.detector = ObjectDetector(config)
        self.config = config

    def process_detections(self, result: 'DetectionResult', img: np.ndarray) -> List[Dict[str, Any]]:
        """Process detection results and filter by size and score"""
        img_width, img_height = img.shape[1], img.shape[0]
        
        # Separate buckets and piles
        pile_boxes, pile_scores, pile_labels = [], [], []
        bucket_boxes, bucket_scores, bucket_labels = [], [], []
        
        for box, score, label in zip(result.boxes, result.scores, result.labels):
            if score < self.config.visualization_score_threshold:
                continue
                
            # Skip if box is too large
            if (box[2]-box[0]) > 0.75*img_width or (box[3]-box[1]) > 0.75*img_height:
                continue
                
            label_text = PILE_QUERIES[label]
            if "bucket" in label_text:
                bucket_boxes.append(box)
                bucket_scores.append(score)
                bucket_labels.append(label)
            else:
                pile_boxes.append(box)
                pile_scores.append(score)
                pile_labels.append(label)

        # Convert to numpy arrays and apply NMS to piles
        detections = []
        if pile_boxes:
            pile_boxes = np.array(pile_boxes)
            pile_scores = np.array(pile_scores)
            pile_labels = np.array(pile_labels)
            pile_boxes, pile_scores, pile_labels = self.detector.apply_nms(
                pile_boxes, pile_scores, pile_labels
            )
            
            # Convert piles to detection format
            for box, score, label in zip(pile_boxes, pile_scores, pile_labels):
                if score < self.config.visualization_score_threshold:
                    continue
                detections.append({
                    'label': PILE_QUERIES[label],
                    'bbox': box.tolist(),
                    'score': float(score)
                })

        # Add bucket detections
        bucket_detections = []
        for box, score, label in zip(bucket_boxes, bucket_scores, bucket_labels):
            bucket_detections.append({
                'label': 'equipment_' + PILE_QUERIES[label],
                'bbox': box.tolist(),
                'score': float(score)
            })

        # Filter out piles that overlap with buckets
        if bucket_detections:
            filtered_detections = []
            for pile in detections:
                keep_pile = True
                pile_box = np.array(pile['bbox'])
                
                for bucket in bucket_detections:
                    bucket_box = np.array(bucket['bbox'])
                    if self.detector.compute_iou(pile_box, bucket_box) > 0.6:
                        keep_pile = False
                        break
                
                if keep_pile:
                    filtered_detections.append(pile)
            
            return filtered_detections
        
        return detections

    def visualize_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Visualize detection results"""
        plt.figure(figsize=(12,8))
        plt.imshow(img)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            label = det['label']
            is_bucket = label.startswith('equipment_')
            
            # Set colors based on type
            color = 'b' if is_bucket else 'r'
            text_color = 'cyan' if is_bucket else 'white'
            
            # Draw bounding box
            rect = plt.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                fill=False,
                linewidth=2,
                edgecolor=color
            )
            plt.gca().add_patch(rect)
            
            # Add score and label
            plt.text(
                x1,
                y1-5,
                f"{score:.2f}",
                color='yellow',
                fontsize=12,
                backgroundcolor='black'
            )
            plt.text(
                x1,
                y1-20,
                label,
                color=text_color,
                fontsize=12,
                backgroundcolor='black'
            )
        
        plt.axis('off')
        plt.show()

    def detect_piles(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect piles in an image and return filtered results"""
        # Load image and run detection
        img = self.detector.load_image(image_path)
        result = self.detector.detect(img, queries=PILE_QUERIES)
        
        # Process and filter detections
        detections = self.process_detections(result, img)
        
        # Visualize if there are any detections
        if detections:
            self.visualize_detections(img, detections)
        
        return detections

def download_videos(url_file: str, video_dir: str) -> None:
    """Download videos from YouTube URLs at 720p"""
    if not os.path.exists(url_file):
        print(f"Error: {url_file} does not exist. Skipping download.")
        return

    os.makedirs(video_dir, exist_ok=True)
    fmt = "bestvideo[height=720]"
    cmd = [
        "yt-dlp",
        "-a", url_file,
        "-f", fmt,
        "--output", os.path.join(video_dir, "%(id)s.%(ext)s")
    ]
    subprocess.run(cmd, check=True)

def extract_frames(video_dir: str, frames_dir: str, fps: int = 1) -> None:
    """Extract frames from videos at specified FPS"""
    if not any(Path(video_dir).glob("*.*")):
        print(f"Error: {video_dir} does not contain any videos. Skipping extraction.")
        return
    
    if any(Path(frames_dir).glob("*.jpg")):
        print(f"Error: {frames_dir} already contains frames. Skipping extraction.")
        return
    
    os.makedirs(frames_dir, exist_ok=True)
    videos = list(Path(video_dir).glob("*.*"))
    
    print(f"\nExtracting frames from {len(videos)} videos:")
    for idx, vidpath in enumerate(videos, 1):
        stem = vidpath.stem
        print(f"\n[{idx}/{len(videos)}] Processing {stem}", flush=True)
        
        out_pattern = os.path.join(frames_dir, f"{stem}_%04d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(vidpath),
            "-vf", f"fps={fps}",
            out_pattern
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ Completed {stem}", flush=True)

def detect_and_save(frames_dir: str, det_dir: str, pipeline: PileDetectionPipeline) -> None:
    """Run detection on frames and save results"""
    if not any(Path(frames_dir).glob("*.jpg")):
        print(f"Error: {frames_dir} does not contain any frames. Skipping detection.")
        return
    
    if any(Path(det_dir).glob("*.json")):
        print(f"Error: {det_dir} already contains detections. Skipping detection.")
        return
    
    os.makedirs(det_dir, exist_ok=True)
    frames = list(Path(frames_dir).glob("*.jpg"))
    total_frames = len(frames)
    
    print(f"\nRunning detection on {total_frames} frames:")
    kept_frames = 0
    deleted_frames = 0
    
    for idx, img_path in enumerate(frames, 1):
        print(f"[{idx}/{total_frames}] Processing {img_path.name}", end="\r", flush=True)
        
        # Run detection
        dets = pipeline.detect_piles(str(img_path))
        
        # Skip if no detections
        if not dets:
            deleted_frames += 1
            continue
            
        kept_frames += 1
        
        # Assign IDs to detections
        dets.sort(key=lambda d: d['bbox'][0])  # Sort by x1 coordinate
        assigned = []
        for i, d in enumerate(dets, start=1):
            x1, y1, x2, y2 = d['bbox']
            assigned.append({
                'id': f'pile{i}',
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            })
        
        # Save JSON
        out_file = os.path.join(det_dir, img_path.name.replace('.jpg', '.json'))
        with open(out_file, 'w') as f:
            json.dump(assigned, f, indent=2)
    
    print(f"\n✓ Completed detection on all {total_frames} frames")
    print(f"  - Kept {kept_frames} frames with pile detections")
    print(f"  - Unused {deleted_frames} frames with no piles", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos, sample frames, detect piles, save JSON per frame"
    )
    parser.add_argument('--urls', default='urls.txt', help='Path to text file with YouTube URLs')
    parser.add_argument('--video_dir', default='videos', help='Directory to save downloaded videos')
    parser.add_argument('--frames_dir', default='frames', help='Directory to save extracted frames')
    parser.add_argument('--det_dir', default='detections', help='Directory to save JSON detections')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second to sample')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='Detection confidence threshold')
    args = parser.parse_args()

    # Initialize pipeline with custom threshold
    config = ModelConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        visualization_score_threshold=args.score_thresh
    )
    pipeline = PileDetectionPipeline(config)

    print("[1/3] Downloading videos...")
    download_videos(args.urls, args.video_dir)

    print("[2/3] Extracting frames...")
    extract_frames(args.video_dir, args.frames_dir, fps=args.fps)

    print("[3/3] Running detection & saving JSON...")
    detect_and_save(args.frames_dir, args.det_dir, pipeline)

    print("Done! JSON files available in:", args.det_dir)

if __name__ == '__main__':
    main()
