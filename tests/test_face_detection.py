from mouseyes.face_detection import FaceDetectionModel
from mouseyes.input_feeder import InputFeeder
from mouseyes.mouseyes_exceptions import NoFaceException
import pytest

DEVICE="CPU"
MODEL='models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
TEST_IMAGE="tests/resources/center.jpg"
TEST_NO_FACE="tests/resources/sofi.jpg"
INPUT_SHAPE=(1, 3, 384, 672)

@pytest.fixture
def model():
    return FaceDetectionModel(MODEL, DEVICE, EXTENSION)

@pytest.fixture
def image():
    ifeed = InputFeeder(TEST_IMAGE)
    return next(ifeed)

@pytest.fixture
def image_no_face():
    ifeed = InputFeeder(TEST_NO_FACE)
    return next(ifeed)

def test_predict_sync(model, image):
    predict(model, image, True)

def test_predict_async(model, image):
    predict(model, image, False)
    
def test_predict_noface(model, image_no_face):
    with pytest.raises(NoFaceException):
        predict(model, image_no_face, True)

def predict(model, image, sync):
    img = model.preprocess_input(image)
    assert img.shape == INPUT_SHAPE
    out = model.predict(img, sync)
    assert out.shape == (1,1,200,7)
    #w = IMAGE_SHAPE[3] # NO: this should be the original image width...
    #h = IMAGE_SHAPE[2] #     ... and height
    w = image.shape[1]
    h = image.shape[0]
    proc_out = model.preprocess_output(out, w, h)
    if proc_out is None:
        raise NoFaceException("There are no faces in this image")
    
    assert proc_out.shape == (1, 4) # only 1 detected face, with 4 points
    assert proc_out[0][0] == 173    # xmin
    assert proc_out[0][1] == 53     # ymin
    assert proc_out[0][2] == 297    # xmax
    assert proc_out[0][3] == 229    # ymax

    cropped_img = model.get_cropped_face(image, proc_out, True)
    assert cropped_img.shape == (176, 124, 3)
