from model import ModelBase

class GazeEstimationModel(ModelBase):
    # gaze-estimation-adas-0002
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    # Inputs:
    #   "left_eye_image": [1x3x60x60], [BxCxHxW], BGR?
    #   "right_eye_image": [1x3x60x60], [BxCxHxW], BGR?
    #   "head_pose_angles": [1x3]
    # Outputs:
    #   "gaze_vector": [1, 3]: Cartesian coordinates of gaze direction vector

    pass