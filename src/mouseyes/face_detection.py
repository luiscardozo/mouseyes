from .model import ModelBase

class FaceDetectionModel(ModelBase):
    # face-detection-adas-binary-0001
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
    # Input: [1x3x384x672], [BxCxHxW], BGR.
    # Outputs: [1, 1, N, 7]
    #                    |--> [image_id, label, conf, x_min, y_min, x_max, y_max]

    def __init__(self, model_path, device='CPU', extensions=None, transpose=(2,0,1)):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose)
        print("Input shape:", super().get_input_shape)
    
