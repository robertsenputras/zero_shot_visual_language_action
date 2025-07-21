import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

# — Hugging-Face Grounding DINO imports —
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# — Efficient-ViT SAM import —
# from efficientvit_sam import EfficientViTSAM

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
        rect = plt.Rectangle((x0,y0), x1-x0, y1-y0,
                             fill=False, linewidth=2, edgecolor='r')
        ax.add_patch(rect)
        ax.text(x0, y0-5, f"{score:.2f}",
                color='yellow', fontsize=12,
                backgroundcolor='black')
    ax.axis('off')
    plt.show()

def grounding_dino_detect(img, hf_model_folder, prompt, device="cpu",
                          box_threshold=0.4, text_threshold=0.3):
    # 1) load processor & model from local HF folder
    processor = AutoProcessor.from_pretrained(hf_model_folder)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
                hf_model_folder).to(device)

    # 2) prepare inputs
    # prompt must be lowercased & end in a dot, e.g. "a cat. a remote control."
    inputs = processor(images=Image.fromarray(img),
                       text=prompt,
                       return_tensors="pt").to(device)

    # 3) forward
    with torch.no_grad():
        outputs = model(**inputs)

    # 4) post-process into boxes/scores
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.shape[:2]]
    )

    # flatten results (assuming single image)
    r = results[0]
    boxes = r["boxes"].cpu().numpy()
    scores = r["scores"].cpu().numpy()
    return boxes, scores


def owl_vit_detect(img, 
                   hf_model="models/owlv2-base-patch16", 
                   queries=["kitten"], 
                   device="cpu", 
                   threshold=0.3):
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
    return boxes, scores, labels


# def efficientvit_sam_mask(img, box, sam_ckpt, device="cpu"):
#     x0,y0,x1,y1 = map(int, box)
#     crop = img[y0:y1, x0:x1]
#     sam = EfficientViTSAM(sam_ckpt, device=device)
#     mask = sam.predict_mask(crop)  # returns H×W numpy array
#     return crop, mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_image("cats.jpeg")
    print("using device: ", device  )
    # # → 1) Grounding DINO
    # prompt = "a kitten."
    # boxes, scores = grounding_dino_detect(
    #     img,
    #     hf_model_folder="models/grounding-dino-tiny",
    #     prompt=prompt,
    #     device=device
    # )
    
    # → 2) Owl-ViT
    boxes, scores, labels = owl_vit_detect(img, queries=["cat"], device=device)


    # → 3) Show boxes
    show_boxes(img, boxes, scores)

    # # → 2) SAM segmentation
    # for i,(box,score) in enumerate(zip(boxes, scores)):
    #     if score < 0.5: continue
    #     crop, mask = efficientvit_sam_mask(
    #         img, box,
    #         sam_ckpt="models/efficientvit_sam.pth",
    #         device=device
    #     )

    #     # visualize
    #     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    #     ax1.imshow(crop); ax1.set_title("Crop"); ax1.axis("off")
    #     ax2.imshow(mask, cmap="gray")
    #     ax2.set_title(f"Mask (score={score:.2f})")
    #     ax2.axis("off")
    #     plt.show()

if __name__=="__main__":
    main()
