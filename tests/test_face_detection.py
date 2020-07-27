from mouseyes.face_detection import FaceDetectionModel
from mouseyes.input_feeder import InputFeeder
import pytest

DEVICE="CPU"
MODEL=f'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
TEST_IMAGE="tests/resources/center.jpg"
IMAGE_SHAPE=(1, 3, 384, 672)

@pytest.fixture
def model():
    return FaceDetectionModel(MODEL, DEVICE, EXTENSION)

@pytest.fixture
def image():
    ifeed = InputFeeder(TEST_IMAGE)
    return next(ifeed)

def test_predict_sync(model, image):
    img = model.preprocess_input(image)
    assert img.shape == IMAGE_SHAPE
    out = model.predict(img, sync=True)
    assert out.shape == (1,1,200,7)
    #w = IMAGE_SHAPE[3] # NO: this should be the original image width...
    #h = IMAGE_SHAPE[2] #     ... and height
    w = image.shape[1]
    h = image.shape[0]
    proc_out = model.preprocess_output(out, w, h)
    assert proc_out.shape == (1, 4) # only 1 detected face, with 4 points
    assert proc_out[0][0] == 173    # xmin
    assert proc_out[0][1] == 53     # ymin
    assert proc_out[0][2] == 297    # xmax
    assert proc_out[0][3] == 229    # ymax

    cropped_img = model.get_cropped_face(image, proc_out, True)
    assert cropped_img.shape == (176, 124, 3)

def test_predict_async(model, image):
    img = model.preprocess_input(image)
    assert img.shape == IMAGE_SHAPE
    out = model.predict(img)
    assert out.shape == (1,1,200,7)
    w = image.shape[1]
    h = image.shape[0]
    proc_out = model.preprocess_output(out, w, h)
    assert proc_out.shape == (1, 4) # only 1 detected face, with 4 points
    assert proc_out[0][0] == 173    # xmin
    assert proc_out[0][1] == 53     # ymin
    assert proc_out[0][2] == 297    # xmax
    assert proc_out[0][3] == 229    # ymax

    cropped_img = model.get_cropped_face(image, proc_out, True)
    assert cropped_img.shape == (176, 124, 3)