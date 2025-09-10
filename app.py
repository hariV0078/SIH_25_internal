from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import cv2
import numpy as np
from dotenv import load_dotenv

# Local imports
from blur import detect_faces, detect_number_plates, blur_regions, check_infrastructure_issues
from map import fetch_important_places, recommend_important_places

load_dotenv()

app = FastAPI(title="SIH_25 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for JSON request body
class LocationRequest(BaseModel):
    lat: float
    lng: float
    radius: int = 2500  # Default radius in meters

@app.get("/", response_class=PlainTextResponse)
def root():
    return "SIH_25 API is running. Endpoints: /blur, /places, /places/markdown"

@app.post("/blur")
async def blur_sensitive_regions_and_detect_issues(file: UploadFile = File(...)):
    """
    Blur sensitive regions (faces and license plates) and detect infrastructure issues.
    Returns JSON response with base64 encoded blurred image and infrastructure detection results.
    """
    tmp_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temporary file
        suffix = os.path.splitext(file.filename or "uploaded.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_bytes = await file.read()
            if len(file_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty file received")
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load image
        image = cv2.imread(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data")

        print(f"Image loaded successfully with shape: {image.shape}")

        # Detect faces - returns numpy array from supervision.Detections.xyxy
        try:
            faces = detect_faces(tmp_path)
            print(f"Face detection completed. Type: {type(faces)}, Shape: {getattr(faces, 'shape', 'N/A')}")
            faces_count = len(faces) if faces is not None and len(faces) > 0 else 0
        except Exception as e:
            print(f"Face detection error: {e}")
            faces = np.array([])  # Empty numpy array to match expected type
            faces_count = 0

        # Detect number plates - returns list of lists
        try:
            plates = detect_number_plates(tmp_path)
            print(f"Plate detection completed. Type: {type(plates)}, Length: {len(plates) if plates else 0}")
            plates_count = len(plates) if plates else 0
        except Exception as e:
            print(f"Plate detection error: {e}")
            plates = []  # Empty list to match expected type
            plates_count = 0

        # Check for infrastructure issues using Gemini (runs automatically)
        try:
            infrastructure_issues = check_infrastructure_issues(tmp_path)
            print(f"Infrastructure issues detection completed: {infrastructure_issues}")
        except Exception as e:
            print(f"Infrastructure issues detection error: {e}")
            infrastructure_issues = {
                'potholes': False,
                'damaged_roads': False,
                'sanitation': False,
                'broken_streetlight': False,
                'water_leakage': False
            }

        # Process the image with the same logic as your working code
        processed_image = image.copy()
        
        # Apply face blurring (faces is numpy array from supervision)
        if faces is not None and len(faces) > 0:
            print(f"Applying blur to {len(faces)} faces")
            processed_image = blur_regions(processed_image, faces)
            print("Face blurring completed successfully")

        # Apply plate blurring (plates is list of lists)  
        if plates and len(plates) > 0:
            print(f"Applying blur to {len(plates)} plates")
            processed_image = blur_regions(processed_image, plates)
            print("Plate blurring completed successfully")

        # Encode the processed image
        success, buffer = cv2.imencode(".jpg", processed_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")

        # Convert image to base64 for JSON response
        import base64
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        print("Image processing completed successfully")
        
        # Count total issues detected
        total_issues = sum(1 for issue in infrastructure_issues.values() if issue)
        
        # Return JSON response with blurred image and infrastructure detection results
        return JSONResponse(content={
            "success": True,
            "message": "Image processed successfully",
            "blurred_image": image_base64,
            "image_format": "jpeg",
            "detections": {
                "faces_detected": faces_count,
                "license_plates_detected": plates_count,
                "faces_blurred": faces_count > 0,
                "plates_blurred": plates_count > 0
            },
            "infrastructure_issues": {
                "potholes": infrastructure_issues.get('potholes', False),
                "damaged_roads": infrastructure_issues.get('damaged_roads', False),
                "sanitation": infrastructure_issues.get('sanitation', False),
                "broken_streetlight": infrastructure_issues.get('broken_streetlight', False),
                "water_leakage": infrastructure_issues.get('water_leakage', False)
            },
            "infrastructure_summary": {
                "total_issues_detected": total_issues,
                "has_any_issues": any(infrastructure_issues.values()),
                "issues_found": [issue.replace('_', ' ').title() for issue, detected in infrastructure_issues.items() if detected]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in blur endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                print(f"Failed to clean up temp file: {e}")

@app.post("/infrastructure-issues")
async def detect_infrastructure_issues_only(file: UploadFile = File(...)):
    """
    Detect infrastructure issues without blurring.
    Returns JSON response with detected issues.
    """
    tmp_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        suffix = os.path.splitext(file.filename or "uploaded.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_bytes = await file.read()
            if len(file_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty file received")
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Check for infrastructure issues using your existing function
        infrastructure_issues = check_infrastructure_issues(tmp_path)
        
        # Count total issues detected
        total_issues = sum(1 for issue in infrastructure_issues.values() if issue)
        
        return {
            "infrastructure_issues": infrastructure_issues,
            "total_issues_detected": total_issues,
            "has_any_issues": any(infrastructure_issues.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in infrastructure-issues endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"Failed to clean up temp file: {e}")

@app.post("/places")
def get_places(request: LocationRequest):
    """
    Get places based on location coordinates sent via JSON request body.
    Expected JSON format:
    {
        "lat": 12.9716,
        "lng": 77.5946,
        "radius": 2500
    }
    """
    try:
        location = f"{request.lat},{request.lng}"
        places = fetch_important_places(location=location, radius=request.radius)
        return JSONResponse(content={
            "location": location, 
            "radius": request.radius, 
            "results": places
        })
    except Exception as e:
        print(f"Error in places endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/places/markdown")
def get_places_markdown(request: LocationRequest):
    """
    Get places in markdown format based on location coordinates sent via JSON request body.
    Expected JSON format:
    {
        "lat": 12.9716,
        "lng": 77.5946,
        "radius": 2500
    }
    """
    try:
        location = f"{request.lat},{request.lng}"
        # recommend_important_places internally calls fetch_important_places
        table = recommend_important_places(location)
        return PlainTextResponse(content=table)
    except Exception as e:
        print(f"Error in places/markdown endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)