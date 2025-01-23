import cv2
import numpy as np
import torch
import dlib
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from .aligner import FaceAligner
def faceDetect(image, facedec):
  image = np.array(image)
  facedec.eval()
  with torch.inference_mode():
      output = facedec(image,verbose=False)
  faces = output[0].boxes.xyxy
  return faces

import numpy as np

def select_best_face(faces, img_center, area_weight=1, distance_weight=1):
  if len(faces) == 0:
        return None, None  
  center_x_img, center_y_img = img_center
  areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])

  center_x_faces = (faces[:, 0] + faces[:, 2]) / 2
  center_y_faces = (faces[:, 1] + faces[:, 3]) / 2

  distances_squared = (center_x_faces - center_x_img) ** 2 + (center_y_faces - center_y_img) ** 2

  scores = area_weight * areas - distance_weight * distances_squared
  best_idx = np.argmax(scores.cpu().numpy())

  selected_face = faces[best_idx]

  return selected_face, best_idx

def faceAlignerPipeline(image, facedec, fa, faces):
  image_np = np.array(image)
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
  x_min, y_min, x_max, y_max = faces.tolist()

  rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
  aligned_face = fa.align(image_np, gray=image_gray, rect=rect)
  faces= faceDetect(aligned_face, facedec).squeeze()
  if len(np.array(faces.cpu().numpy()).shape) > 1:
    width, height = image.size
    img_center = (width / 2, height / 2)
    faces, selected_face_idx = select_best_face(faces, img_center)
  return aligned_face,faces

def croppingFace(aligned_face, faces):
  x_min, y_min, x_max, y_max = faces.tolist()
  cropped_face = aligned_face[int(y_min):int(y_max), int(x_min):int(x_max)]
  return cropped_face

def faceEmbedder(model, cropped_face, preprocess):
  preprocessed = preprocess(cropped_face)
  model.eval()
  with torch.inference_mode():
    embedding = model(preprocessed.unsqueeze(0))
  return embedding

def who_is_it(db_vec, img_vec, threshold=1):
    if not db_vec or img_vec is None:
        print("Cơ sở dữ liệu rỗng hoặc vector ảnh đầu vào không hợp lệ.")
        return False

    min_distance = float('inf')
    closest_name = None

    for name, enc in db_vec.items():
        distance = np.linalg.norm(enc - img_vec)

        if distance < min_distance:
            min_distance = distance
            closest_name = name

    if min_distance >= threshold:
        print(f"Không có người nào trong cơ sở dữ liệu với khoảng cách nhỏ hơn {threshold}.")
        return False

    return closest_name

