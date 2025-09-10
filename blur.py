import cv2
from PIL import Image
import base64
import os
import requests
import torch
from google import genai
from google.genai import types
from dotenv import load_dotenv
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

# -----------------------------
# Helper Functions
# -----------------------------

def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# -----------------------------
# Face & Number Plate Detection
# -----------------------------

def detect_faces(image_path):
    """Detects faces using a YOLOv8 model from Hugging Face."""
    print("Detecting faces using Hugging Face model...")
    try:
        model_path = hf_hub_download(repo_id="HariVaradhan/YOLOv8-Face-Detection", filename="model.pt")
        model = YOLO(model_path)
        output = model(Image.open(image_path))
        results = Detections.from_ultralytics(output[0])
        faces = results.xyxy  # List of bounding boxes in (x1, y1, x2, y2) format
        return faces
    except Exception as e:
        print(f"An error occurred during face detection: {e}")
        return []

def detect_number_plates(image_path):
    """Detects number plates using a YOLOS model from Hugging Face."""
    print("\nDetecting number plates using Hugging Face model...")
    try:
        image = Image.open(image_path).convert("RGB")
        feature_extractor = YolosFeatureExtractor.from_pretrained('HariVaradhan/license-plate-detection')
        model = YolosForObjectDetection.from_pretrained('HariVaradhan/license-plate-detection')

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        
        plates = [box.tolist() for box in results["boxes"]]
        return plates
    except Exception as e:
        print(f"An error occurred during number plate detection: {e}")
        return []

def blur_regions(image, regions):
    """Applies a Gaussian blur to specified regions of an image."""
    for (x1, y1, x2, y2) in regions:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue

        kernel_w = max(1, (x2 - x1) // 2)
        kernel_h = max(1, (y2 - y1) // 2)
        if kernel_w % 2 == 0: kernel_w += 1
        if kernel_h % 2 == 0: kernel_h += 1
        
        blurred_roi = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image

# -----------------------------
# Infrastructure Issues Detection using Gemini
# -----------------------------

def check_infrastructure_issues(image_path):
    """Checks for various infrastructure issues in an image using the Gemini API."""
    print("\nChecking for infrastructure issues using Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return {
            'potholes': False,
            'damaged_roads': False,
            'sanitation': False,
            'broken_streetlight': False,
            'water_leakage': False
        }
        
    try:
        base64_image = encode_image(image_path)
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash-exp"

        # Define the prompt for multiple infrastructure issues
        prompt = """
        Analyze this image for the following infrastructure issues and respond with only the categories found:

        Categories to check:
        1. Potholes - holes or depressions in road surface
        2. Damaged Roads - cracks, broken pavement, deteriorated road surface
        3. Sanitation - garbage, litter, waste, dirty areas, overflowing bins
        4. Broken Streetlight - damaged, fallen, or non-functional street lighting
        5. Water Leakage - visible water leaks, broken pipes, water accumulation from infrastructure

        Respond in this exact format for each category found:
        POTHOLES: YES/NO
        DAMAGED_ROADS: YES/NO
        SANITATION: YES/NO
        BROKEN_STREETLIGHT: YES/NO
        WATER_LEAKAGE: YES/NO

        Only respond with the above format, nothing else.
        """

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(base64_image),
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
        ):
            if hasattr(chunk, 'text'):
                full_response += chunk.text

        print(f"Gemini response: {full_response}")

        # Parse the response
        results = {
            'potholes': False,
            'damaged_roads': False,
            'sanitation': False,
            'broken_streetlight': False,
            'water_leakage': False
        }

        # Extract results from response
        response_upper = full_response.upper()
        
        if "POTHOLES: YES" in response_upper:
            results['potholes'] = True
        if "DAMAGED_ROADS: YES" in response_upper:
            results['damaged_roads'] = True
        if "SANITATION: YES" in response_upper:
            results['sanitation'] = True
        if "BROKEN_STREETLIGHT: YES" in response_upper:
            results['broken_streetlight'] = True
        if "WATER_LEAKAGE: YES" in response_upper:
            results['water_leakage'] = True

        return results
            
    except Exception as e:
        print(f"An error occurred while checking for infrastructure issues: {e}")
        return {
            'potholes': False,
            'damaged_roads': False,
            'sanitation': False,
            'broken_streetlight': False,
            'water_leakage': False
        }

# -----------------------------
# Main Execution
# -----------------------------

def main():
    load_dotenv()
    
    image_path = "inputt.jpg"
    output_path = "blurred_output.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Input image not found at '{image_path}'")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from '{image_path}'")
        return

    # Detect and blur faces using Hugging Face model
    faces = detect_faces(image_path)
    print(f"Detected {len(faces)} faces.")
    image = blur_regions(image, faces)

    # Detect and blur number plates using Hugging Face model
    plates = detect_number_plates(image_path)
    print(f"Detected {len(plates)} number plates.")
    image = blur_regions(image, plates)

    # Save blurred image
    cv2.imwrite(output_path, image)
    print(f"\nBlurred image saved as '{output_path}'")

    # Check for infrastructure issues using Gemini
    infrastructure_issues = check_infrastructure_issues(image_path)
    print("\nInfrastructure Issues Detected:")
    for issue, detected in infrastructure_issues.items():
        status = "YES" if detected else "NO"
        print(f"{issue.replace('_', ' ').title()}: {status}")

if __name__ == "__main__":
    main()