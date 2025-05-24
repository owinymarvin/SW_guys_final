import cv2
import torch
import numpy as np
import time
import json
import urllib

from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize
from pytorchvideo.transforms import UniformTemporalSubsample

class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

print("Loading SlowFast model...")
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
device = "cpu"
model = model.eval()
model = model.to(device)
print("Model loaded successfully.")

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.request.urlretrieve(json_url, json_filename)
except Exception as e:
    print(f"Failed to download class names: {e}")
    kinetics_id_to_classname = {0: "Unknown", 1: "Still"}
    print("Using dummy class names.")

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {int(v): k for k, v in kinetics_classnames.items()}

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
clip_duration_seconds = (num_frames * sampling_rate) / frames_per_second

transform_pipeline = Compose([
    Lambda(lambda x: x / 255.0),
    Normalize(mean, std),
    PackPathway()
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure it's not in use by another application.")
    exit()

webcam_fps = cap.get(cv2.CAP_PROP_FPS)
if webcam_fps == 0 or webcam_fps > 60:
    webcam_fps = 30.0
print(f"Webcam FPS: {webcam_fps}")

frames_needed_for_raw_clip = num_frames * sampling_rate
print(f"Will collect {frames_needed_for_raw_clip} raw frames for a clip, then subsample to {num_frames} frames for prediction.")

frame_buffer = []
current_prediction = "No action detected yet..."
last_prediction_time = time.time()
prediction_interval = 3

print("\nStarting webcam feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    frame_tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
    
    spatial_transform = Compose([
        Resize(side_size),
        CenterCrop(crop_size)
    ])
    
    frame_tensor = spatial_transform(frame_tensor)

    frame_buffer.append(frame_tensor)

    if len(frame_buffer) > frames_needed_for_raw_clip:
        frame_buffer.pop(0)

    if time.time() - last_prediction_time > prediction_interval and len(frame_buffer) == frames_needed_for_raw_clip:
        
        input_raw_clip_tensor = torch.stack(frame_buffer, dim=1)
        
        # Original PyTorchVideo models often expect (T, C, H, W) for certain transforms.
        # Although the model's final input is (N, C, T, H, W), an intermediate permutation
        # might be what's expected for `Normalize` in combination with `PackPathway`.
        # This permutation was previously removed, but the error indicates it's still needed.
        subsampled_clip = UniformTemporalSubsample(num_frames)(input_raw_clip_tensor).permute(1, 0, 2, 3) # C,T,H,W -> T,C,H,W
        
        try:
            # Apply the rest of the transforms (normalization, PackPathway).
            # The `transform_pipeline` expects (C, T, H, W) for `PackPathway`.
            # So, we need to permute back after Normalize if Normalize operated on (T, C, H, W).
            # Let's adjust the transform pipeline to handle the (T, C, H, W) input after subsampling.

            # Re-define a temporary transform pipeline to handle the (T, C, H, W) input
            temp_transform_pipeline = Compose([
                Lambda(lambda x: x / 255.0),
                # Normalize operates on (C, H, W) by default, or (N, C, H, W).
                # When given (T, C, H, W), it will normalize each 'frame' (T) across its channels (C).
                Normalize(mean, std),
                # Now permute to (C, T, H, W) for PackPathway
                Lambda(lambda x: x.permute(1, 0, 2, 3)),
                PackPathway()
            ])
            processed_input = temp_transform_pipeline(subsampled_clip)


            inputs_with_batch_dim = [i.to(device)[None, ...] for i in processed_input]

            with torch.no_grad():
                preds = model(inputs_with_batch_dim)

            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=1).indices[0]

            if int(pred_classes[0]) in kinetics_id_to_classname:
                 current_prediction = kinetics_id_to_classname[int(pred_classes[0])]
            else:
                 current_prediction = f"Unknown action ID: {int(pred_classes[0])}"
            print(f"Predicted: {current_prediction}")

        except Exception as e:
            current_prediction = f"Error during prediction: {e}"
            print(f"Prediction Error: {e}")
        
        last_prediction_time = time.time()

    display_frame = frame.copy()

    cv2.putText(display_frame, f"Action: {current_prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('SlowFast Webcam Demo (CPU)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam demo finished.")