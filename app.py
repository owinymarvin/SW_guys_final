import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import threading
from collections import Counter
from huggingface_hub import hf_hub_download

# >>>>>> CRITICAL: Ensure these imports are present and correct <<<<<<
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize
from pytorchvideo.transforms import UniformTemporalSubsample
from flask import Flask, jsonify
# >>>>>> CRITICAL: End of imports <<<<<<

# --- Configuration for loading your fine-tuned model from Hugging Face Hub ---
HF_MODEL_REPO_ID = "owinymarvin/slowfast-violence-detector-finetuned"
LOCAL_CACHE_DIR = "./hf_model_cache"

# Create cache directory if it doesn't exist
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# --- Download model files from Hugging Face Hub ---
print(f"Downloading model files from Hugging Face Hub: {HF_MODEL_REPO_ID}...")
try:
    model_weights_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename="pytorch_model.bin",
        cache_dir=LOCAL_CACHE_DIR
    )
    model_config_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename="config.json",
        cache_dir=LOCAL_CACHE_DIR
    )
    print(f"Model weights downloaded to: {model_weights_path}")
    print(f"Model config downloaded to: {model_config_path}")
except Exception as e:
    print(f"Error downloading model files from Hugging Face Hub: {e}")
    print("Please ensure the model repository exists and is accessible.")
    exit() # Exit if we can't download the model

# --- Load custom class names and model configuration ---
id_to_label = {0: "non_violence", 1: "violence"} # Default in case config.json fails
NUM_LABELS = 2 # Default

if os.path.exists(model_config_path):
    try:
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        id_to_label = {int(k): v for k, v in model_config.get("id2label", {}).items()}
        NUM_LABELS = model_config.get("num_labels", NUM_LABELS)
        print(f"Loaded custom labels from downloaded config.json: {id_to_label}")
    except Exception as e:
        print(f"Warning: Could not load config.json ({e}). Using default labels: {id_to_label}")
else:
    print(f"Warning: config.json not found at {model_config_path}. Using default labels: {id_to_label}")

# --- Define global video transformation parameters (consistent with training) ---
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32 # Number of frames for SlowFast input
sampling_rate = 2
frames_per_second = 30 # Assuming a typical webcam FPS, adjust if known
slowfast_alpha = 4 # Alpha for SlowFast pathway split

# --- Define PackPathway class (must be consistent with training) ---
class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, frames: torch.Tensor):
        # frames is (C, T, H, W)
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1, # Select along the time dimension
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

# --- Model Loading ---
print("Loading SlowFast model architecture...")
try:
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    print("Base slowfast_r50 architecture loaded successfully.")
except Exception as e:
    print(f"Error loading base SlowFast model from PyTorch Hub: {e}")
    print("Please ensure you have an active internet connection or the model is cached.")
    exit()

# Reinitialize the final classification head to match your custom classes
try:
    original_classifier_layer = model.blocks[-1].proj
    print(f"Original classifier layer found at model.blocks[-1].proj")
except AttributeError:
    print("Attempting to find classifier layer at model.head.proj (fallback)...")
    try:
        original_classifier_layer = model.head.proj
        print(f"Original classifier layer found at model.head.proj")
    except AttributeError:
        print("Could not find the classification head. Exiting.")
        exit()

new_classifier_layer = nn.Linear(original_classifier_layer.in_features, NUM_LABELS)

if hasattr(model.blocks[-1], 'proj'):
    model.blocks[-1].proj = new_classifier_layer
elif hasattr(model.head, 'proj'):
    model.head.proj = new_classifier_layer

print(f"Model's classification head reinitialized for {NUM_LABELS} classes.")

# Load your fine-tuned state_dict from the downloaded file
try:
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print(f"Fine-tuned weights loaded from {model_weights_path}.")
except Exception as e:
    print(f"Error loading fine-tuned weights from {model_weights_path}: {e}")
    # The warning below indicates it will proceed, but it's important to note if weights fail
    print("Proceeding with the base pre-trained model (not fine-tuned). This might indicate a problem with the downloaded weights.")

# Set model to evaluation mode
model.eval()

# Move model to appropriate device (CPU for local testing as requested, or GPU if available)
device = "cpu" # Set to "cuda" if you have a GPU and want to use it
model = model.to(device)
print(f"Model moved to {device}.")


# Calculate how many raw frames we need to collect to get `num_frames` after subsampling
frames_needed_for_raw_clip = num_frames * sampling_rate

# Define the full transformation pipeline for a single video tensor
transform_pipeline = Compose([
    Lambda(lambda x: x / 255.0), # Normalize pixel values to [0, 1]
    # Permute to (T, C, H, W) for Normalize to apply correctly along channel dim (index 1)
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    Normalize(mean, std),        # Normalizes across channels (C=3)
    # Permute back to (C, T, H, W) for PackPathway
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    PackPathway()                # Split into slow and fast pathways
])

# --- Global variables for API to access ---
# Initialize with a default state that clearly indicates no prediction yet
current_prediction_data = {"prediction": "Waiting for first aggregated prediction...", "confidence": 0.0, "timestamp": time.time()}
prediction_lock = threading.Lock() # To safely update prediction data

# --- Flask API Setup ---
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Returns the latest aggregated violence detection prediction from the model.
    """
    with prediction_lock: # Safely read the global shared variable
        return jsonify(current_prediction_data)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Violence Detection API</h1><p>Access /predict for the latest aggregated prediction.</p>"

# --- Function to run the Flask app in a separate thread ---
def run_flask_app():
    print("\nStarting Flask API in background thread...")
    # use_reloader=False is crucial to prevent the app from starting twice in development mode
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- Main Webcam Processing and Prediction Loop (runs in the main thread) ---
if __name__ == '__main__':
    # Start the Flask API in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    cap = cv2.VideoCapture(0) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure it's not in use by another application or try a different index (e.g., 1).")
        # Exit if webcam cannot be opened, as the main functionality depends on it
        exit()

    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    if webcam_fps == 0 or webcam_fps > 60: # Fallback if FPS is not reliably reported
        webcam_fps = 30.0
    print(f"Webcam detected FPS: {webcam_fps:.2f}")
    print(f"Will collect {frames_needed_for_raw_clip} raw frames for a clip for each prediction.")

    frame_buffer = [] # Stores raw frames (as tensors) for the current clip
    display_prediction_text = "Initializing..." # Text displayed on OpenCV window

    # --- Intervals for model prediction and API reporting ---
    current_prediction_interval = 1 # Model predicts every 1 second (adjust for more/less frequent internal checks)
    api_reporting_interval = 10 # API updates every 10 seconds with aggregated result

    last_model_prediction_time = time.time()
    last_api_report_time = time.time()

    # Buffer for predictions within the api_reporting_interval
    short_interval_prediction_buffer = [] # Stores (label, confidence) for aggregation

    print("\nStarting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Convert frame from BGR (OpenCV) to RGB, then to PyTorch tensor (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        
        # Apply spatial transforms (Resize, CenterCrop) to the current frame
        spatial_transform_per_frame = Compose([
            Resize(side_size),
            CenterCrop(crop_size)
        ])
        frame_tensor = spatial_transform_per_frame(frame_tensor)

        frame_buffer.append(frame_tensor)

        # Maintain a buffer of `frames_needed_for_raw_clip` frames
        if len(frame_buffer) > frames_needed_for_raw_clip:
            frame_buffer.pop(0)

        # --- Perform model prediction at `current_prediction_interval` ---
        # Only predict if enough frames are collected for a full clip
        if time.time() - last_model_prediction_time > current_prediction_interval and len(frame_buffer) == frames_needed_for_raw_clip:
            
            # Stack buffered frames into a single tensor (C, T, H, W)
            input_raw_clip_tensor = torch.stack(frame_buffer, dim=1)

            # Apply temporal subsampling to get the required number of frames for the model
            subsampled_clip = UniformTemporalSubsample(num_frames)(input_raw_clip_tensor)
            
            # Apply the rest of the transforms (normalization, PackPathway)
            processed_input = transform_pipeline(subsampled_clip)

            # Add batch dimension and move to device
            inputs_with_batch_dim = [i.to(device)[None, ...] for i in processed_input]

            try:
                with torch.no_grad():
                    preds = model(inputs_with_batch_dim)

                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(preds)
                pred_classes = preds.topk(k=1).indices[0]

                predicted_id = int(pred_classes[0])
                predicted_label = id_to_label.get(predicted_id, f"Unknown ID: {predicted_id}")
                confidence = preds[0, predicted_id].item()
                
                # Add current prediction to the short-interval buffer for aggregation
                short_interval_prediction_buffer.append((predicted_label, confidence))
                # print(f"Model predicted (1s interval): {predicted_label} (Conf: {confidence:.2f})") # Uncomment for verbose internal predictions

            except Exception as e:
                print(f"Prediction Error at {current_prediction_interval}s interval: {e}")
                # Even if there's an error, log it and add a placeholder to buffer
                short_interval_prediction_buffer.append(("error", 0.0))
            
            last_model_prediction_time = time.time()

        # --- Aggregate and update API at `api_reporting_interval` ---
        # This block runs every 10 seconds to process the buffered predictions
        if time.time() - last_api_report_time > api_reporting_interval:
            
            if short_interval_prediction_buffer:
                # Perform aggregation (majority vote for label, average confidence for that label)
                labels_only = [p[0] for p in short_interval_prediction_buffer if p[0] not in ["error"]] # Exclude errors from majority vote
                
                if labels_only:
                    most_common_label = Counter(labels_only).most_common(1)[0][0] # Majority vote for the label
                    
                    # Calculate average confidence *for the most common label*
                    confidences_for_majority = [p[1] for p in short_interval_prediction_buffer if p[0] == most_common_label]
                    aggregated_confidence = sum(confidences_for_majority) / len(confidences_for_majority) if confidences_for_majority else 0.0
                    
                    aggregated_prediction_label = most_common_label
                    aggregated_confidence_value = round(aggregated_confidence, 4)
                else: # This path is hit if the buffer only contains "error" entries or is empty of valid predictions
                    aggregated_prediction_label = "No valid predictions in interval"
                    aggregated_confidence_value = 0.0
                    
                print(f"\n--- Aggregated Prediction for API (every {api_reporting_interval}s): {aggregated_prediction_label} (Avg Conf: {aggregated_confidence_value:.2f}) ---\n")

                # Update the global shared variable, accessible by the Flask API
                with prediction_lock:
                    current_prediction_data = {
                        "prediction": aggregated_prediction_label,
                        "confidence": aggregated_confidence_value,
                        "timestamp": time.time()
                    }
                
                # Update the text displayed on the OpenCV window
                display_prediction_text = f"{aggregated_prediction_label} (Avg. Conf: {aggregated_confidence_value:.2f})"

                # Clear the buffer for the next aggregation window
                short_interval_prediction_buffer = []

            else:
                # This case handles scenarios where no predictions could be made in the last interval (e.g., webcam issues)
                aggregated_prediction_label = "No predictions in last interval"
                aggregated_confidence_value = 0.0
                print(f"\n--- Aggregated Prediction for API (every {api_reporting_interval}s): {aggregated_prediction_label} ---\n")
                with prediction_lock:
                    current_prediction_data = {
                        "prediction": aggregated_prediction_label,
                        "confidence": aggregated_confidence_value,
                        "timestamp": time.time()
                    }
                display_prediction_text = aggregated_prediction_label


            last_api_report_time = time.time() # Reset timer for the next 10-second aggregation window

        # Display the frame with the current aggregated prediction
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Action: {display_prediction_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('SlowFast Webcam Demo', display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam demo finished.")
    print("Flask API thread will terminate when main program exits.")