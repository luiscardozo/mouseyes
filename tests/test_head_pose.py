from mouseyes.head_pose_estimation import HeadPoseEstimationModel
from mouseyes.input_feeder import InputFeeder
import pytest
import numpy as np

DEVICE="CPU"
PRECISION="FP32"
MODEL=f'models/intel/head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
TEST_IMAGE="tests/resources/cropped_face.jpg"
PITCH=5.169604
ROLL=-0.56816363
YAW=-6.813993
INPUT_SHAPE=(1, 3, 60, 60)

@pytest.fixture
def model():
    return HeadPoseEstimationModel(MODEL, DEVICE, EXTENSION)

@pytest.fixture
def image():
    ifeed = InputFeeder(TEST_IMAGE)
    return next(ifeed)

@pytest.fixture
def outputs():
    # simulate the output of HeadPoseEstimationModel.predict()
    return {
        "angle_p_fc": np.array([[PITCH]]),
        "angle_r_fc": np.array([[ROLL]]),
        "angle_y_fc": np.array([[YAW]]),
    }

def test_head_pose(model, image):
    img = model.preprocess_input(image)
    assert img.shape == INPUT_SHAPE
    out = model.predict(img, True)
    print(out)
    assert "angle_p_fc" in out
    assert "angle_r_fc" in out
    assert "angle_y_fc" in out
    pitch = out['angle_p_fc']
    roll = out['angle_r_fc']
    yaw = out['angle_y_fc']
    assert pitch.shape == (1,1)
    assert roll.shape == (1,1)
    assert yaw.shape == (1,1)

    assert pitch[0] == PITCH
    assert roll[0] == ROLL
    assert yaw[0] == YAW

def test_head_pose_async(model, image):
    img = model.preprocess_input(image)
    assert img.shape == INPUT_SHAPE
    out = model.predict(img) #only returns 1 output, in this case is better to get every output separately
    #pitch = model.get_output(0, "angle_p_fc")
    #roll = model.get_output(0, "angle_r_fc")
    #yaw = model.get_output(0, "angle_y_fc")
    assert "angle_p_fc" in out
    assert "angle_r_fc" in out
    assert "angle_y_fc" in out
    pitch = out['angle_p_fc']
    roll = out['angle_r_fc']
    yaw = out['angle_y_fc']

    assert pitch.shape == (1,1)
    assert roll.shape == (1,1)
    assert yaw.shape == (1,1)

    assert pitch[0] == PITCH
    assert roll[0] == ROLL
    assert yaw[0] == YAW

def test_postprocess(model, outputs):
    # Gaze pide "np.array([YAW, PITCH, ROLL])"
    assert "angle_p_fc" in outputs
    assert "angle_r_fc" in outputs
    assert "angle_y_fc" in outputs

    pitch = outputs['angle_p_fc']
    roll = outputs['angle_r_fc']
    yaw = outputs['angle_y_fc']

    assert pitch.shape == (1,1)
    assert roll.shape == (1,1)
    assert yaw.shape == (1,1)

    assert pitch[0] == PITCH
    assert roll[0] == ROLL
    assert yaw[0] == YAW

    out = model.preprocess_output(outputs)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(YAW)
    assert out[1] == pytest.approx(PITCH)
    assert out[2] == pytest.approx(ROLL)
