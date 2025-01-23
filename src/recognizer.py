from ultralytics import YOLO
from models.Inception_ResnetV1.InceptionResnetV1 import InceptionResnetV1
from utils.helper import faceDetect, select_best_face, faceAlignerPipeline, croppingFace, faceEmbedder
from utils.aligner import FaceAligner
import torch
import torchvision.transforms as T
import numpy as np
import dlib
import matplotlib.pyplot as plt

class ImageEncodePipeline(object):
    def __init__(self, detector_path,predictor_path, device = "cuda" if torch.cuda.is_available() else "cpu", preprocess=None):
        self.detector = YOLO(detector_path).to(device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval()
        if preprocess is None:
            preprocess= T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor()
            ])
        self.preprocess = preprocess
        self.predictor = dlib.shape_predictor(predictor_path)

    def encode(self, image):
        image = image.resize((640, 640))
        faces = faceDetect(image, self.detector)
        if len(faces) == 0:
            return None
        # x_min, y_min, x_max, y_max = faces[0].tolist()
        # image_cv = np.array(image)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        # cv2.rectangle(image_cv, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.show()
        width, height = image.size
        img_center = (width / 2, height / 2)
        selected_face, selected_face_idx = select_best_face(faces, img_center)
        if selected_face is None:
            return None
        # x_min, y_min, x_max, y_max = selected_face.tolist()
        # image_cv = np.array(image)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        # cv2.rectangle(image_cv, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.show()
        fa = FaceAligner(self.predictor, desiredFaceWidth=640)
        aligned_face, faces = faceAlignerPipeline(image, self.detector, fa, selected_face)
        if faces is None:
            return
        # x_min, y_min, x_max, y_max = faces
        # image_cv = np.array(aligned_face)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        # cv2.rectangle(image_cv, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.show()
        cropped_face = croppingFace(aligned_face, faces)
        # plt.imshow(cropped_face)
        # plt.axis('off')
        # plt.show()
        em_vector = faceEmbedder(self.embedder,cropped_face, self.preprocess)
        return em_vector

