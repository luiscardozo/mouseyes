from mouseyes.model import ModelBase
import numpy as np

YAW="angle_y_fc"
PITCH="angle_p_fc"
ROLL="angle_r_fc"

class HeadPoseEstimationModel(ModelBase):
    # head-pose-estimation-adas-0001
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    # Input: [1x3x60x60], [1xCxHxW], BGR
    # YAW [-90,90], PITCH [-70,70], ROLL [-70,70]
    #
    # Outputs:
    #   "angle_y_fc": [1, 1]: yaw (in degrees).
    #   "angle_p_fc": [1, 1]: pitch (in degrees).
    #   "angle_r_fc": [1, 1]: roll (in degrees).

    def predict(self, image, sync=False, request_id=0):
        super().predict(image, sync, request_id)
        output = {
            "angle_p_fc": self.get_output(request_id, "angle_p_fc"),
            "angle_r_fc": self.get_output(request_id, "angle_r_fc"),
            "angle_y_fc": self.get_output(request_id, "angle_y_fc"),
        }
        return output

    def preprocess_output(self, outputs):
        """
        Postprocess the outputs for the usage in Gaze Estimation.
        format: np.array [YAW, PITCH, ROLL]. Shape: (3, )
        """
        return np.array([outputs[YAW], outputs[PITCH], outputs[ROLL]]).flatten()
