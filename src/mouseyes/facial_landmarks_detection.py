from mouseyes.model import ModelBase
import numpy as np

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
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = out[0]
        x1 = x1[0][0]
        y1 = y1[0][0]
        x2 = x2[0][0]
        y2 = y2[0][0]
        x3 = x3[0][0]
        y3 = y3[0][0]
        x4 = x4[0][0]
        y4 = y4[0][0]
        x5 = x5[0][0]
        y5 = y5[0][0]
        return np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5])

    def get_cropped_eyes(self, original_img, coords, save_file=False):
        """Returns a np with the cropped eyes, in original image's size"""
        
        right_eye = original_img[coords[1]:coords[3], coords[0]:coords[2]]
        left_eye = original_img[coords[1]:coords[3], coords[0]:coords[2]]
        if save_file:
            self._save_image(right_eye, "right-eye")
            self._save_image(left_eye, "left-eye")
        return right_eye, left_eye

    def _save_image(self, frame, name):
        img_path = f'/home/luis/tmp/{name}.jpg'    #for debug only. Change to your own dir
        print("write ", img_path)
        cv2.imwrite(img_path, frame)
        return img_path
