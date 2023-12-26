from imutils import face_utils
import dlib

def find_eye_corners(image, landmark_path, ):
  p = landmark_path
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(p)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)
  for (i, rect) in enumerate(rects):

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    return shape[36], shape[39], shape[42], shape[45]
