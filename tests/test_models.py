import pytest
from mouseyes.model import ModelBase
from mouseyes.face_detection import FaceDetectionModel
from mouseyes.facial_landmarks_detection import FacialLandmarksModel
from mouseyes.head_pose_estimation import HeadPoseEstimationModel
from mouseyes.gaze_estimation import GazeEstimationModel
from mouseyes.mouseyes_exceptions import ModelXmlFileNotFoundException, ModelBinFileNotFoundException


DEVICE="CPU" #I have only a 3rd gen i7.
PRECISION="FP16"
VALID_MODEL=f'models/intel/landmarks-regression-retail-0009/{PRECISION}/landmarks-regression-retail-0009.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

def test_load_existent_model_with_xml_ext():
    "Load existent, valid model, with XML extension. Should load OK (or throw UnsupportedLayersException if MYRIAD)"
    ModelBase(VALID_MODEL, DEVICE)

def test_load_existent_model_without_xml_ext():
    "Load existent, valid model, without XML extension. Should load OK (or throw UnsupportedLayersException if MYRIAD)"
    no_xml_model = f'models/intel/landmarks-regression-retail-0009/{PRECISION}/landmarks-regression-retail-0009'
    ModelBase(no_xml_model, DEVICE)

def test_load_existent_model_xml_without_bin():
    "Load existent model with xml file, but without bin file. Should raise ModelBinFileNotFoundException"
    xml_path = 'models/nobinmodel/nobinmodel.xml'
    with pytest.raises(ModelBinFileNotFoundException):
        ModelBase(xml_path, DEVICE)

def test_load_nonexistent_model_raises_exception():
    "Try to load non-existent model. Should raise ModelXmlFileNotFoundException"
    xml_path = 'nonexistent'
    with pytest.raises(ModelXmlFileNotFoundException):
        fd = FaceDetectionModel(xml_path, DEVICE)

def test_load_face_detection():
    "Load Face Detection Model"
    xml_path = 'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
    FaceDetectionModel(xml_path, DEVICE, EXTENSION)

def test_load_face_landmarks():
    "Load Face Landmarks Model"
    xml_path = f'models/intel/landmarks-regression-retail-0009/{PRECISION}/landmarks-regression-retail-0009.xml'
    FacialLandmarksModel(xml_path, DEVICE)

def test_load_head_pose():
    "Load Head Pose Estimation Model"
    xml_path = f"models/intel/head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml"
    HeadPoseEstimationModel(xml_path, DEVICE)

def test_load_gaze():
    "Load Gaze Model"
    xml_path = f"models/intel/gaze-estimation-adas-0002/{PRECISION}/gaze-estimation-adas-0002.xml"
    GazeEstimationModel(xml_path, DEVICE, EXTENSION)