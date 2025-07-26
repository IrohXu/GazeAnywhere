from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection


def ovod_head_detection(image_path, processor, model):
    image = Image.open(image_path)
    text_labels = [["head of a child"]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.tensor([(image.height, image.width)])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    # Retrieve predictions for the first image for the corresponding text queries
    result = results[0]
    boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
    output_boxes = []
    for box, score, text_label in zip(boxes, scores, text_labels):
        box = [round(i, 2) for i in box.tolist()]
        output_boxes.append((round(score.item(), 3), box))

    output_boxes.sort(key=lambda x: x[0], reverse=True)
    if output_boxes != []:
        return output_boxes[0][1]
    else:
        return None
    
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
image_path = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/WeillCornell_PedabyteProject_processed/for_annotations/all/PWC007_2019_02_23_T1_panasonic_ESCS_merged_12129_15508/clipped_frames/0720.jpg"

print(ovod_head_detection(image_path, processor, model))