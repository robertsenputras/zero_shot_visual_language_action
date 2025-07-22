# scrape_yt.py
# Implements YouTube video scraping and frame extraction for construction site footage.
# Uses yt-dlp for reliable video downloading and OpenCV for frame processing.

import os
import cv2
import json
import argparse
from typing import List, Dict
from datetime import datetime
from yt_dlp import YoutubeDL
from pathlib import Path

class VideoScraper:
    """
    YouTube video scraper specialized for construction site footage.
    Handles video downloading, frame extraction, and dataset organization.
    """
    
    def __init__(self, output_dir: str = "dataset/raw_frames"):
        """
        Initializes the video scraper with output directory configuration.
        
        Args:
            output_dir: Base directory for storing downloaded frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Limit resolution for efficiency
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s')
        }

    def download_video(self, url: str) -> str:
        """
        Downloads a single video from YouTube.
        
        Args:
            url: YouTube video URL
            
        Returns:
            str: Path to downloaded video file
        """
        with YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            return video_path

    def extract_frames(
        self,
        video_path: str,
        interval: int = 30,
        min_brightness: float = 40.0,
        min_contrast: float = 10.0
    ) -> List[str]:
        """
        Extracts frames from video at specified interval with quality checks.
        
        Args:
            video_path: Path to input video file
            interval: Frame extraction interval in frames
            min_brightness: Minimum average brightness threshold
            min_contrast: Minimum contrast threshold
            
        Returns:
            List[str]: Paths to extracted frame images
        """
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        video_id = Path(video_path).stem
        frames_dir = self.output_dir / video_id
        frames_dir.mkdir(exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Quality checks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                contrast = gray.std()
                
                if brightness > min_brightness and contrast > min_contrast:
                    frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
            
            frame_count += 1
        
        cap.release()
        return frame_paths

    def process_video_list(self, video_urls: List[str], metadata_file: str = "dataset_info.json"):
        """
        Processes a list of videos and saves extraction metadata.
        
        Args:
            video_urls: List of YouTube video URLs
            metadata_file: Path to save dataset metadata
        """
        dataset_info = {
            "creation_date": datetime.now().isoformat(),
            "videos": []
        }
        
        for url in video_urls:
            try:
                print(f"Processing: {url}")
                video_path = self.download_video(url)
                frame_paths = self.extract_frames(video_path)
                
                video_info = {
                    "url": url,
                    "frames_extracted": len(frame_paths),
                    "frame_paths": frame_paths
                }
                dataset_info["videos"].append(video_info)
                
                # Clean up downloaded video
                os.remove(video_path)
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        # Save dataset metadata
        with open(self.output_dir / metadata_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)

def main():
    """Main execution function for video scraping pipeline."""
    parser = argparse.ArgumentParser(description="YouTube construction site video scraper")
    parser.add_argument(
        "--urls",
        required=True,
        help="Path to text file containing YouTube URLs (one per line)"
    )
    parser.add_argument(
        "--output",
        default="dataset/raw_frames",
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Frame extraction interval"
    )
    args = parser.parse_args()

    # Read video URLs
    with open(args.urls) as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    # Initialize and run scraper
    scraper = VideoScraper(args.output)
    scraper.process_video_list(video_urls)

if __name__ == "__main__":
    main()
