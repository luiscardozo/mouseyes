from mouseyes.model import ModelBase
import numpy as np
import cv2

DEFAULT_DEVICE="CPU"
DEFAULT_MODEL=f'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
DEFAULT_EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
DEFAULT_TRANSPOSE=(2,0,1)

class FaceDetectionModel(ModelBase):
    # face-detection-adas-binary-0001
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
    # Input: [1x3x384x672], [BxCxHxW], BGR.
    # Outputs: [1, 1, N, 7]
    #                    |--> [image_id, label, conf, x_min, y_min, x_max, y_max]

    def __init__(self, model_path=DEFAULT_MODEL, device=DEFAULT_DEVICE, extensions=DEFAULT_EXTENSION, transpose=DEFAULT_TRANSPOSE):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose)
        print("Input shape:", super().get_input_shape)
    
    def get_cropped_face(self, original_img, processed_output, save_file=False):
        """Returns a np with the cropped face, in original image's size"""
        coords=processed_output[0]
        cropped_face = original_img[coords[1]:coords[3], coords[0]:coords[2]]
        #cropped_face = original_img
        #xmin, ymin, xmax, ymax = processed_output[0]
        #cv2.rectangle(cropped_face, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        if save_file:
            self._save_image(cropped_face)
        return cropped_face

    def _save_image(self, frame):
        img_path = f'/home/luis/tmp/cropped_face.jpg'    #for debug only. Change to your own dir
        print("write ", img_path)
        cv2.imwrite(img_path, frame)
        return img_path