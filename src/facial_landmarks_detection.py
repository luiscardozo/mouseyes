from model import ModelBase

class FacialLandmarksModel(ModelBase):
    # landmarks-regression-retail-0009
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    # Input: [1x3x48x48], [BxCxHxW], BGR
    # Outputs: [1, 10]
    #               |--> (x0, y0, x1, y1, ..., x5, y5)  (range[0,1])
    # 
    # From the eye positions, extract and return the images for left and right eyes
    pass