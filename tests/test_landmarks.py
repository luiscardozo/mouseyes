from mouseyes.facial_landmarks_detection import FacialLandmarksModel
from mouseyes.input_feeder import InputFeeder
import pytest

DEVICE="CPU"
PRECISION="FP32"
MODEL=f'models/intel/landmarks-regression-retail-0009/{PRECISION}/landmarks-regression-retail-0009.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
TEST_IMAGE="tests/resources/cropped_face.jpg"
INPUT_SHAPE=(1, 3, 48, 48)

@pytest.fixture
def model():
    return FacialLandmarksModel(MODEL, DEVICE, EXTENSION)

@pytest.fixture
def image():
    ifeed = InputFeeder(TEST_IMAGE)
    return next(ifeed)

def test_landmarks(model, image):
    predict(model, image, True)

def test_landmarks_async(model, image):
    predict(model, image, False)

def predict(model, image, sync):
    img = model.preprocess_input(image)
    assert img.shape == INPUT_SHAPE
    out = model.predict(img, sync)

    assert out.shape == (10, )

    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = out
    assert x1 == pytest.approx(0.2390861)
    assert x3 == pytest.approx(0.42645836)
    assert y5 == pytest.approx(0.7644643)
