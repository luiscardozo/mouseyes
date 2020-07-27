from .model import ModelBase

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
    
    def __init__(self, model_path, device='CPU', extensions=None, transpose=(2,0,1)):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose)

    
