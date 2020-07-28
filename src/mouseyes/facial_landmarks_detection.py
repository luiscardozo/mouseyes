from mouseyes.model import ModelBase
import numpy as np
import cv2

class FacialLandmarksModel(ModelBase):
    # landmarks-regression-retail-0009
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    # Input: [1x3x48x48], [BxCxHxW], BGR
    # Outputs: [1, 10]
    #               |--> (x0, y0, x1, y1, ..., x5, y5)  (range[0,1])
    # 
    # From the eye positions, extract and return the images for left and right eyes
    
    def __init__(self, model_path, device='CPU', extensions=None, transpose=(2,0,1)):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose)
    
    def predict(self, image, sync=False, request_id=0):
        out = super().predict(image, sync=sync, request_id=request_id)
        #x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = out[0]
        return out.flatten()

    #Copied from https://knowledge.udacity.com/questions/283702
    def preprocess_output(self, results, image):
        height, width = image.shape[0:2]
        #print(f"Output Height: {height} :: Width: {width}")
        coordinates = []
        for i in range(0,4,2):
            point = (int(results[i]*width), int(results[i+1]*height))
            xmin = point[0]-20
            ymin = point[1]-20
            xmax = point[0]+20
            ymax = point[1]+20
            coordinates.append((xmin, ymin, xmax, ymax))
        return coordinates

    def get_cropped_eyes(self, frame, landmarks, save_file=False):
        """Returns a np with the cropped eyes, from original image's size"""
        coords = self.preprocess_output(landmarks, frame)
        eye_nr=1
        eyes = []
        for box in coords:
            #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cropped_eye = frame[box[1]:box[3], box[0]:box[2]]            
            eye_nr+=1
            if save_file:
                self._save_image(cropped_eye, f"eye{eye_nr}")
            if cropped_eye.shape == (40,40,3):              #  ToDo: see this. It' very specific for the size given by preprocess_output()
                eyes.append(cropped_eye)
        
        eyes = np.array(eyes)
        return eyes

    def _save_image(self, frame, name):
        img_path = f'/home/luis/tmp/{name}.jpg'    #for debug only. Change to your own dir
        print("write ", img_path)
        cv2.imwrite(img_path, frame)
        return img_path
