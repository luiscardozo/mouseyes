from mouseyes.face_detection import FaceDetectionModel
from mouseyes.input_feeder import InputFeeder
import pytest

DEVICE="CPU"
MODEL=f'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
TEST_IMAGE="tests/resources/center.jpg"

@pytest.fixture
def model():
    return FaceDetectionModel(MODEL, DEVICE, EXTENSION)

@pytest.fixture
def image():
    ifeed = InputFeeder(TEST_IMAGE)
    return next(ifeed)

def test_predict_sync(model, image):
    img = model.preprocess_input(image)
    output = model.predict(img, sync=True)
    assert output['detection_out'].shape == (1,1,200,7)
    