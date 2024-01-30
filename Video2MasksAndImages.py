import numpy as np
import cv2
import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import csv
from concurrent.futures import ThreadPoolExecutor

def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab

device = torch.device("cpu")
deeplab = make_deeplab(device)

deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_deeplab(deeplab, img, device):
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15)

# Open a video file for reading
cap = cv2.VideoCapture('./vd/3.mp4')

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set the desired resolution (you can adjust these values)
target_width = 640
target_height = 480

# Create folders to save images and masks
os.makedirs('mydata6/images', exist_ok=True)
os.makedirs('mydata6/masks', exist_ok=True)

frame_id = 0
frame_interval = 1  # Process every 20th frame

# Create a list to store image and mask paths
image_mask_paths = []

# Function to process frames in parallel
def process_frame(frame_id, frame):
    # Resize the frame to the target resolution
    frame = cv2.resize(frame, (target_width, target_height))

    # Apply the DeepLabV3 model to generate the mask for persons
    mask = apply_deeplab(deeplab, frame, device)

    # Save the image
    image_filename = os.path.join('mydata6/images', f'image_{frame_id:04d}.jpg')
    cv2.imwrite(image_filename, frame)

    # Save the mask
    mask_filename = os.path.join('mydata6/masks', f'mask_{frame_id:04d}.png')
    cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)

    # Append image and mask paths to the list
    image_mask_paths.append({
        'id': frame_id,
        'images': image_filename,
        'masks': mask_filename
    })

with ThreadPoolExecutor() as executor:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            executor.submit(process_frame, frame_id, frame)

        frame_id += 1

# Release the video objects
cap.release()
cv2.destroyAllWindows()

# Save the image and mask paths in a CSV file
csv_filename = 'mydata6/image_mask_paths.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['id', 'images', 'masks']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(image_mask_paths)

print(f'CSV file saved as {csv_filename}')
