from model import ModelBase

class HeadPoseEstimationModel(ModelBase):
    # head-pose-estimation-adas-0001
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    # Input: [1x3x60x60], [1xCxHxW], BGR
    # Outputs:
    #   "angle_y_fc": [1, 1]: yaw (in degrees).
    #   "angle_p_fc": [1, 1]: pitch (in degrees).
    #   "angle_r_fc": [1, 1]: roll (in degrees).
    
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device, extensions)