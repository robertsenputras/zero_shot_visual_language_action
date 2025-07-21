import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from torchvision.ops import nms

# — Hugging-Face Grounding DINO imports —
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_image(path_or_url):
    if path_or_url.startswith("http"):
        img = Image.open(requests.get(path_or_url, stream=True).raw)
        return np.array(img)
    else:
        img = cv2.imread(path_or_url)[:, :, ::-1]
        return img

def show_boxes(img, boxes, scores):
    fig,ax = plt.subplots(1, figsize=(12,8))
    ax.imshow(img)
    for (x0,y0,x1,y1),score in zip(boxes,scores):
        if (score < 0.2):
            continue
        rect = plt.Rectangle((x0,y0), x1-x0, y1-y0,
                             fill=False, linewidth=2, edgecolor='r')
        ax.add_patch(rect)
        ax.text(x0, y0-5, f"{score:.2f}",
                color='yellow', fontsize=12,
                backgroundcolor='black')
    ax.axis('off')
    plt.show()


def compute_iou(box1, boxes):
    """Compute IoU between a box and an array of boxes"""
    # Calculate intersection coordinates
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    # Calculate area of intersection
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate area of boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    return iou

def compute_nms(boxes, scores, iou_threshold=0.3, sigma=0.5):
    """
    Apply Fast NMS (Soft-NMS) to remove overlapping boxes
    Args:
        boxes: numpy array of shape (N, 4) containing bounding boxes
        scores: numpy array of shape (N,) containing confidence scores
        iou_threshold: IoU threshold for considering boxes as overlapping
        sigma: parameter for gaussian penalty function
    Returns:
        filtered boxes, scores, and indices
    """
    if len(boxes) == 0:
        return boxes, scores, np.array([], dtype=np.int32)
    
    # Convert to numpy if they're torch tensors
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Initialize lists for keeping track of boxes to keep
    indices = np.arange(len(scores))
    updated_scores = scores.copy()
    
    # Sort boxes by score
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    updated_scores = updated_scores[order]
    indices = indices[order]
    
    # Apply Soft-NMS
    for i in range(len(boxes)):
        # Get IoU of box i with all remaining boxes
        ious = compute_iou(boxes[i], boxes[i+1:])
        
        # Apply gaussian penalty to overlapping boxes
        overlapping = ious > iou_threshold
        updated_scores[i+1:][overlapping] *= np.exp(-(ious[overlapping]**2)/sigma)
    
    # Keep boxes above score threshold
    keep = updated_scores > 0.001  # Small threshold to remove very low scoring boxes
    filtered_boxes = boxes[keep]
    filtered_scores = updated_scores[keep]
    filtered_indices = indices[keep]
    
    # Sort by updated scores
    order = np.argsort(filtered_scores)[::-1]
    
    return filtered_boxes[order], filtered_scores[order], filtered_indices[order]

def owl_vit_detect(img, 
                   hf_model="models/owlv2-base-patch16", 
                   queries=["kitten"], 
                   device="cpu", 
                   threshold=0.22,
                   nms_threshold=0.3):
    # 1) load
    processor = AutoProcessor.from_pretrained(hf_model)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                    hf_model).to(device)

    # 2) prepare inputs (note: list of queries)
    inputs = processor(
        text=queries,
        images=Image.fromarray(img),
        return_tensors="pt"
    ).to(device)

    # 3) inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4) post-process into boxes/scores/labels
    target_sizes = torch.tensor([img.shape[:2]], device=device)
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes
    )[0]

    boxes  = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    print("before nms: ", scores, " boxes: ", boxes)
    # 5) Apply NMS
    if len(boxes) > 0:
        boxes, scores, keep_indices = compute_nms(boxes, scores, nms_threshold)
        labels = labels[keep_indices]
    
    print("after nms: ", scores, " boxes: ", boxes)
    return boxes, scores, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_image("piles_2.png")
    print("using device: ", device)

    queries = [
        # generic “pile” to catch anything
        "pile",  "stockpile",  "heap",

        # sand/fines
        "pile of sand", "sand heap", "sand stockpile",

        # gravel / fine aggregate
        "gravel pile", "heap of gravel", "gravel stockpile", "aggregate pile",

        # crushed stone / rock
        "crushed rock pile", "stone pile", "rock heap",

        # generic earth
        "earth mound", "soil mound",

        # (optional) catch-all “spoil” jargon
        "spoil pile", "spoil heap",
    ]

    # → 2) Owl-ViT
    boxes, scores, labels = owl_vit_detect(img, 
                                           queries = queries,
                                           device=device)

    # → 3) Show boxes
    show_boxes(img, boxes, scores)

if __name__=="__main__":
    main()
