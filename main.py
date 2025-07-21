import torch
from model import ObjectDetector
from config import ModelConfig, PILE_QUERIES

def main():
    # Initialize config and detector
    config = ModelConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    detector = ObjectDetector(config)
    
    # Load image
    img = detector.load_image("piles_2.png")
    print("using device: ", detector.device)

    # Run detection
    result = detector.detect(img, queries=PILE_QUERIES)

    # Visualize results
    detector.visualize(img, result)

if __name__=="__main__":
    main()
