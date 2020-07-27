from mouseyes.model import ModelBase

DEFAULT_DEVICE="CPU"
DEFAULT_MODEL=f'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
DEFAULT_EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
DEFAULT_TRANSPOSE=(2,0,1)

class FaceDetectionModel(ModelBase):
    # face-detection-adas-binary-0001
    # https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
    # Input: [1x3x384x672], [BxCxHxW], BGR.
    # Outputs: [1, 1, N, 7]
    #                    |--> [image_id, label, conf, x_min, y_min, x_max, y_max]

    def __init__(self, model_path=DEFAULT_MODEL, device=DEFAULT_DEVICE, extensions=DEFAULT_EXTENSION, transpose=DEFAULT_TRANSPOSE):
        super().__init__(model_path, device=device, extensions=extensions, transpose=transpose)
        print("Input shape:", super().get_input_shape)
    

if __name__ == '__main__':
    #A little "inline" test
    fd = FaceDetectionModel()
    from mouseyes.input_feeder import InputFeeder
    TEST_IMAGE = "tests/resources/center.jpg"
    ifeed = InputFeeder('image', TEST_IMAGE)
    ifeed.load_data()
    print(type(ifeed.cap))
    print(ifeed.cap.shape)
