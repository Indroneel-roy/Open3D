import numpy as np
import cv2
import time
import os
from datetime import datetime

path_model = "models/"

# Create output directory for saved images
output_dir = "saved_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created {output_dir} directory for saving images")

# Check if models directory exists
if not os.path.exists(path_model):
    os.makedirs(path_model)
    print(f"Created {path_model} directory")
    print("Please download the model files:")
    print("Large model: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx")
    print("Small model: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx")
    exit()

# Read Network
model_name = "model-f6b98070.onnx"  # MiDaS v2.1 Large
# model_name = "model-small.onnx"  # MiDaS v2.1 Small (uncomment to use small model)

model_path = path_model + model_name

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    print("Please download the model file:")
    if "f6b98070" in model_name:
        print("Large model: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx")
    else:
        print("Small model: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx")
    exit()

# Load the DNN model
model = cv2.dnn.readNet(model_path)

if (model.empty()):
    print("Could not load the neural net! - Check path")
    exit()

print("Model loaded successfully!")

# Set backend and target - use CPU since CUDA not available
print("Using CPU backend")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Controls:")
print("- Press 's' to save current image and depth map")
print("- Press 'q' to quit")

# Counter for saved images
save_counter = 0

while cap.isOpened():
    # Read in the image
    success, img = cap.read()
    
    if not success:
        print("Failed to read from webcam")
        break

    imgHeight, imgWidth, channels = img.shape

    # start time to calculate FPS
    start = time.time()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create Blob from Input Image
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    if "f6b98070" in model_name:
        blob = cv2.dnn.blobFromImage(img_rgb, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)
    else:
        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(img_rgb, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()
    
    output = output[0,:,:]
    output = cv2.resize(output, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    
    # Show FPS and instructions
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Press 's' to save, 'q' to quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Original Image', img)
    cv2.imshow('Depth Map', depth_map)

    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original image
        img_filename = f"{output_dir}/image_{timestamp}_{save_counter:03d}.jpg"
        cv2.imwrite(img_filename, img)
        
        # Save depth map (convert to 8-bit for saving)
        depth_8bit = (depth_map * 255).astype(np.uint8)
        depth_filename = f"{output_dir}/depth_{timestamp}_{save_counter:03d}.jpg"
        cv2.imwrite(depth_filename, depth_8bit)
        
        # Also save depth map as numpy array for later use
        depth_npy_filename = f"{output_dir}/depth_{timestamp}_{save_counter:03d}.npy"
        np.save(depth_npy_filename, depth_map)
        
        print(f"Saved: {img_filename}")
        print(f"Saved: {depth_filename}")
        print(f"Saved: {depth_npy_filename}")
        
        save_counter += 1

cap.release()
cv2.destroyAllWindows()

print(f"Total images saved: {save_counter}")
print(f"Images saved in: {output_dir}/")