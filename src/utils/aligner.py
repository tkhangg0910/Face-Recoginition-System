from collections import OrderedDict
import numpy as np
import cv2
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords
import numpy as np
import cv2
class FaceAligner:
  def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
    desiredFaceWidth=256, desiredFaceHeight=None):
    # store the facial landmark predictor, desired output left
    # eye position, and desired output face width + height
    self.predictor = predictor
    self.desiredLeftEye = desiredLeftEye
    self.desiredFaceWidth = desiredFaceWidth
    self.desiredFaceHeight = desiredFaceHeight

    # if the desired face height is None, set it to be the
    # desired face width (normal behavior)
    if self.desiredFaceHeight is None:
      self.desiredFaceHeight = self.desiredFaceWidth

  def align(self, image, gray, rect):
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = self.predictor(gray, rect)
    shape = shape_to_np(shape)

    #simple hack ;)
    if (len(shape)==68):
      # extract the left and right eye (x, y)-coordinates
      (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
      (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    else:
      (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
      (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
    desiredDist *= self.desiredFaceWidth
    scale = desiredDist / dist
    scale *= 0.75
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
      (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = self.desiredFaceWidth * 0.5
    tY = self.desiredFaceHeight * self.desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    # apply the affine transformation
    (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
    # (h, w) = image.shape[:2]
    output =cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    # x_min, y_min, x_max, y_max = rect.left(), rect.top(), rect.right(), rect.bottom()

    # # Apply the affine transform to rect corners
    # corners = np.array([[x_min, y_min, 1], [x_max, y_max, 1]])
    # transformed_corners = np.dot(M, corners.T).T

    # new_x_min, new_y_min = transformed_corners[0][:2]
    # new_x_max, new_y_max = transformed_corners[1][:2]
    # aligned_rect = (
    #     int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)
    # )
    # return the aligned face
    return output