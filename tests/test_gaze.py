from mouseyes.gaze_estimation import GazeEstimationModel
from mouseyes.input_feeder import InputFeeder
import pytest
import numpy as np

DEVICE="CPU"
PRECISION="FP32"
MODEL=f'models/intel/gaze-estimation-adas-0002/{PRECISION}/gaze-estimation-adas-0002.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
RIGHT_EYE_IMAGE="tests/resources/left_eye.jpg"
LEFT_EYE_IMAGE="tests/resources/right_eye.jpg"
#INPUT_SHAPE=(1, 3, 48, 48)
PITCH=5.169604
ROLL=-0.56816363
YAW=-6.813993

@pytest.fixture
def model():
    return GazeEstimationModel(MODEL, DEVICE, EXTENSION)

def image(path):
    ifeed = InputFeeder(path)
    return next(ifeed)

@pytest.fixture
def left_eye():
    return image(LEFT_EYE_IMAGE)

@pytest.fixture
def right_eye():
    return image(RIGHT_EYE_IMAGE)

@pytest.fixture
def head_pose():
    return np.array([YAW, PITCH, ROLL])

def test_gaze(model, left_eye, right_eye, head_pose):
    predict_gaze(model, left_eye, right_eye, head_pose, True)

def test_gaze_async(model, left_eye, right_eye, head_pose):
    predict_gaze(model, left_eye, right_eye, head_pose, False)

def predict_gaze(model,left_eye, right_eye, head_pose, sync):
    
    prep_left, prep_right, prep_hp = model.preprocess_input(left_eye, right_eye, head_pose)
    
    #assert img.shape == INPUT_SHAPE
    out = model.predict(prep_left, prep_right, prep_hp, sync)

    assert out.shape == (1, 3)
    print(out)
    assert out[0][0] == pytest.approx(-0.04466014)
    assert out[0][1] == pytest.approx(0.090213664)
    assert out[0][2] == pytest.approx(-0.98609734)

