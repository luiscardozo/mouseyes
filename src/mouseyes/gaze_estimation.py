from mouseyes.model import ModelBase
import numpy as np
import time

class GazeEstimationModel(ModelBase):
    # gaze-estimation-adas-0002
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    # Inputs:
    #   "left_eye_image": [1x3x60x60], [BxCxHxW], BGR?
    #   "right_eye_image": [1x3x60x60], [BxCxHxW], BGR?
    #   "head_pose_angles": [1x3]
    # Outputs:
    #   "gaze_vector": [1, 3]: Cartesian coordinates of gaze direction vector

    def __init__(self, model_path, device='CPU', extensions=None, transpose=(2,0,1), logger=None):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose, logger=logger)

    def preprocess_input(self, left_eye, right_eye, head_pose):
        prep_left = super().preprocess_input(left_eye, required_size=None, input_name="left_eye_image")
        prep_right = super().preprocess_input(right_eye, required_size=None, input_name="right_eye_image")
        prep_head_pose = np.array([head_pose])
        return (prep_left, prep_right, prep_head_pose)

    def predict(self, left_eye, right_eye, head_pose, sync=False, request_id=0):
        '''
        This method is meant for running predictions on the input images.
        '''
        input_blob = {
            "head_pose_angles": head_pose,
            "left_eye_image": left_eye,
            "right_eye_image": right_eye
        }

        if sync:
            startt = time.perf_counter()
            out = self.infer_sync(input_blob)
            super().log(f"**************** Time to infer model (sync) (gaze_estimation): {time.perf_counter()-startt}")
            return out[self.output_name]        #normally returns a dict
        else:
            startt = time.perf_counter()
            self.infer_async(input_blob, request_id)
            if self.wait(request_id) == 0:
                super().log(f"**************** Time to infer model (async+wait) (gaze_estimation): {time.perf_counter()-startt}")
                return self.get_output(request_id)

    def preprocess_output(self, outputs):
        return outputs.flatten()