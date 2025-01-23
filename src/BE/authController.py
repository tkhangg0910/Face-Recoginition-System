from fastapi import FastAPI, File, UploadFile, Path, HTTPException
import sys
import os
from typing import List, Dict, Any
from pydantic import BaseModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from recognizer import ImageEncodePipeline
from contextlib import asynccontextmanager
from utils.helper import who_is_it
from pymilvus import utility
from db_config import get_collection
from BE.authService import insert_user, search_user, show_data
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path
import numpy as np
import base64

db= None
encoder = None
BASE_DIR = Path(__file__).resolve().parent.parent
@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, db
    encoder = ImageEncodePipeline(f"{BASE_DIR}/models/YOLOV11_Face/yolov11m-face.pt", f"{BASE_DIR}/models/Landmark_predictor/shape_predictor_68_face_landmarks.dat")
    db = get_collection()
    print(db.schema)
    print("Model and db loaded successfully.")
    yield
    encoder = None
    db = None
    print("Model and db unloaded successfully.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def decode_base64_image(file: str):
    try:
        if file.startswith("data:image/"):
            header, file_content = file.split(",", 1) 
            padding = len(file_content) % 4
            if padding != 0:
                file_content += '=' * (4 - padding)
            return base64.b64decode(file_content)
        else:
            raise HTTPException(status_code=400, detail="Invalid base64 string.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding base64: {str(e)}")
    
@app.post("/login")
async def login(payload: Dict[Any, Any]):
    file = payload.get("image")
    try:
        file_content = decode_base64_image(file)
        image = Image.open(io.BytesIO(file_content))
        image = image.convert("RGB")
        en_vec = encoder.encode(image)
        if en_vec is not None:
            name = search_user(en_vec.squeeze().numpy().astype(np.float32).tolist(), db)
            if name:
                return {"name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding image: {str(e)}")
    return {"name": None}


@app.post("/register")
async def register(payload: Dict[Any, Any]):
    name = payload.get("name")
    files = payload.get("files")
    if len(files) != 3:
        return {"error": "You must upload exactly 3 images."}

    encoded_vectors = []
    for file in files:
        try:
            # Reading the binary content of the file
            file_content = decode_base64_image(file)

            # Convert the binary content to a PIL Image
            image = Image.open(io.BytesIO(file_content)) 
            image = image.convert("RGB")
            en_vec = encoder.encode(image)
            if en_vec is not None:
                encoded_vectors.append(en_vec.squeeze().numpy().astype(np.float32).tolist())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")
    if len(encoded_vectors) == 3:
        # Insert the user data, assuming encoded_vectors are processed correctly
        data = [{"name": name, "embedding": encoded_vectors[i]} for i in range(3)]

        print("inserting")
        success = insert_user(data, db)
        print("inserted")
        if success:
            show_data(db)
            return {"name": name, "status": "registered successfully"}
        else:
            return {"error": "Error inserting user data."}
    else:
        return {"error": "Failed to encode all images."}
    
# class RegisterModel(BaseModel):
#     name: str
#     files: List[UploadFile]
    
# @app.post("/register")
# async def register(register_data: RegisterModel):
#     global encoder
#     print(register_data)
#     name = register_data.name
#     files = register_data.files

#     # Ensure exactly 3 images are uploaded
#     if len(files) != 3:
#         return {"error": "You must upload exactly 3 images."}

#     encoded_vectors = []
#     for file in files:
#         try:
#             en_vec = encoder.encode(file.file)
#             if en_vec is not None:
#                 encoded_vectors.append(en_vec)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

#     if len(encoded_vectors) == 3:
#         data = [
#             [name] * 3, 
#             encoded_vectors
#         ]
        
#         success = insert_user(data, db)
        
#         if success:
#             show_data(db)
#             return {"name": name, "status": "registered successfully"}
#         else:
#             return {"error": "Error inserting user data."}
#     else:
#         return {"error": "Failed to encode all images."}