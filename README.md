# Computer Pointer Controller

This is the AI Eyes-Controlled Computer Mouse App!
Use your webcam (or some video) to move your mouse pointer!
(But remember that the webcam is like an reversed mirror. If you move to the right, it will move to your left).

## Project Set Up and Installation

First, I suppose that you already have OpenVINO installed. If not, please [download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) and [install](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) it. You will need a 6th to 10th generation Intel CPU.
You also need Python 3 installed.

Then, you should get the code. For this, please clone this repository to your computer. I am assuming you're using Linux (in Windows, the commands should be somewhat similar, but I cannot gide you exactly).
**Clone the repository:**
```
cd to_your_directory
git clone https://github.com/luiscardozo/mouseyes
cd mouseyes
```

### Download the pretrained models
Supposing you already have OpenVINO installed with all it's dependencies, run the following to download the required pretrained models:
```
/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001,head-pose-estimation-adas-0001,landmarks-regression-retail-0009,gaze-estimation-adas-0002 -o models/
```
(Run this outside of the virtual environment --created below--, as it does not have all the required dependencies of the Model Downloader)

### Install the Virtual Environment
Create the virtual environment and install project dependencies.
```
virtualenv env
#add OpenVINO initialization to environment initialization
echo "source /opt/intel/openvino/bin/setupvars.sh" >> env/bin/activate
source env/bin/activate
pip install -r requirements.txt
```

Intall the package in editable mode (required for pytest)
```
python3 setup.py install
pip install -e .
```
The installation and configuration only needs to be done once.
The next commands need to be inside the virtual environment. If you got off of the virtual environment (via the `deactivate` command or by closing the terminal), you can gain access to the virtual environment with `source env/bin/activate` (inside mouseyes directory).

### Test the code
```
python -m pytest
```

## Demo
To run this code, please run (inside the virtual environment, in the root folder):
```
python3 main.py
```
There are no required arguments (it will use some defaults). You can get a list of possible arguments with `python3 main.py --help`.
For example, with no argument, the program will try to use your webcam (the same as `python3 main.py -i cam` ). If you want to use a video file, you can try passing a video path to the -i option (for example: `python3 main.py -i test/resources/video.webm`).

## Documentation
When running without any arguments, it will show a window with your face (or the contents of the video). If you want to hide it, you should run with `--hide_window`
To **stop** the program, press `ESC` or `q`, or hit `Ctrl+C` in the terminal.
It uses predefined models in the models/ directory. To change the path of the models, or to try with another models (with similar outpus), you can change with the `--face_model`, `--head_pose_model`, `--landmarks_model` and `--gaze_model`.
You can run in CPU, GPU, VPU (for example, Neural Compute Stick 2 or NCS2. Device option: "MYRIAD") or FPGA, passing one of these options to `--device`.

If you want to move the mouse within the same main thread, use `--same_thread`. Otherwise, a secondary thread will move the mouse, making the video or webcam more fluid, but picking just some of the "move" commands.

### Directory structure
```
mouseyes/                           <== root directory
├── main.py                         <== entry point to execute the program
├── README.md                       <== The file that you are reading right now
├── requirements.txt                <== Dependencies listing to install with pip -r requirements.txt
├── setup.py                        <== To install the package (python3 setup.py install)
├── models                          <== You can put your pretrained models here
│   ├── info.txt                    <== Information on how to download pretrained models from Intel Model Zoo
│   └── nobinmodel                  <== Specific model directory
│       └── nobinmodel.xml          <== Model with no .bin file (for testing purposes)
├── src                             <== Source directory
│   └── mouseyes                    <== mouseyes package
│       ├── __init__.py             <== this file marks this directory as a Python Package
│       ├── main.py                 <== Main file. This is where the major logic is performed
│       ├── model.py                <== ModelBase class. All models inherit from this. It calls the functions of OpenVINO
│       ├── face_detection.py       <== Face Detection Model class. Inherits from ModelBase
│       ├── facial_landmarks_detection.py   <== Facial Landmarks Detection Model class. Inherits from ModelBase
│       ├── head_pose_estimation.py         <== Head Pose Estimation model class. Inherits from ModelBase
│       ├── gaze_estimation.py              <== Gaze Estimation Model class. Inherits from ModelBase
│       ├── input_feeder.py                 <== Controls the input (video, webcam, image file)
│       ├── mouse_controller.py             <== Controls the mouse
│       └── mouseyes_exceptions.py          <== Custom Exceptions for this project
├── bin
│   └── demo.mp4                            <== Original video for demo
└── tests                                   <== Tests to see if everything is working
    ├── __init__.py
    ├── resources                           <== Resources for the tests
    │   ├── center.jpg
    │   ├── cropped_face.jpg
    │   ├── ...
    │   ├── video.webm
    │   └── wink_left.jpg
    ├── test_face_detection.py              <== Test files
    ├── test_gaze.py
    ├── test_head_pose.py
    ├── test_image_feeder.py
    ├── test_landmarks.py
    └── test_models.py
```

## Benchmarks
### Testing with other hardwares
As I have only a 3rd gen i7 (OpenVINO officially supports only 6th to 10th gen Intel Processors), I had to use CPU extensions in order to be able to run the system in my CPU.
Trying to run the software in GPU, gives me an error: `clGetPlatformIDs error -1001`
Trying to run in NCS2 (MYRIAD) gives me `unsupported layer type "Quantize"`
Trying to run in `HETERO:MYRIAD,CPU` or `HETERO:GPU,CPU`, gives me an error because I need to apply the CPU extension but HETERO cannot accept the extension.

So, for now, I can only test in CPU with extensions, until I get a newer hardware.

### Execution Times Results

Running with different model precisions and measuring time:

#### Test 1:
* Description: "Default" precisions: Face Detection in INT1 (face-detection-adas-binary-0001), other models in FP32
* Command: `python3 main.py --dev --logfile tests/results/mouseyes.default.log`
* Log: `tests/results/mouseyes.default.log`
* Method to calculate the values:


**load time:**
* face: 0.1864
* head: 0.0512
* landmarks: 0.0342
* gaze: 0.5088

**infer time:**
* face infer (async): ~0.0306
* head pose (sync): ~0.0036
* landmarks (sync): ~0.0011
* gaze (sync): around: ~0.0046


#### Test 2:
* Description: All the models in FP32 (Face Detection: face-detection-adas-0001)
* Command: `python3 main.py --dev --face_model models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml --logfile tests/results/mouseyes.fp32_all.log`
* Log: `tests/results/mouseyes.fp32_all.log`

**load time:**
* face: 0.1985
* head: 0.0519
* landmarks: 0.0405
* gaze: 0.0701

**infer time:**
* face infer (async): ~0.0644
* head pose (sync): ~0.0047
* landmarks (sync): ~0.0015
* gaze (sync): around: ~0.0075


#### Test 3:
* Description: All models in FP16 (Face Detection: face-detection-adas-0001)
* Command: `python3 main.py --dev --face_model models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml --head_pose_model models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml --landmarks_model models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml --gaze_model models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml --logfile tests/results/mouseyes.fp16.log`
* Log: `tests/results/mouseyes.fp16.log`

**load time:**
* face: 0.2110
* head: 0.0755
* landmarks: 0.0415
* gaze: 0.0838

**infer time:**
* face infer (async): ~0.0647
* head pose (sync): ~0.0046
* landmarks (sync): ~0.0015
* gaze (sync): around: ~0.0075

#### Test 4:
* Description: Face and Gaze in INT8. Others in FP32. (Face Detection: face-detection-adas-0001)
* Command: `python3 main.py --dev --face_model models/intel/face-detection-adas-0001/INT8/face-detection-adas-0001.xml --gaze_model models/intel/gaze-estimation-adas-0002/INT8/gaze-estimation-adas-0002.xml --logfile tests/results/mouseyes.int8.log`
* Log: `tests/results/mouseyes.int8.log`

**load time:**
* face: 0.1439
* head: 0.0542
* landmarks: 0.0395
* gaze: 0.0801

**infer time:**
* face infer (async): ~0.0503
* head pose (sync): ~0.0048
* landmarks (sync): ~0.0015
* gaze (sync): around: ~0.0052


To calculate the mean values from the log files, I created a avg.py that receives all the values and calculate the mean.
For example, to extract "face detection infer time" on mouseyes.fp32_all.log and calculate the mean:
```
$ cd tests/results
$ grep "Time to infer model (async+wait) (face-detection-adas-0001" mouseyes.fp32_all.log | cut -d: -f5 | python3 avg.py
Items:  (135,)
Mean:  0.06441624244438976
Max:  0.07500506799988216
Min:  0.06196728700160747
```

### Async Inference
Using sync vs async, I've got almost the same time. That is because (I think), to fully leverage the power of async inference, I need to continue with the next inferences instead of waiting for the result of the inference that I recently sent to the OpenVINO engine. This should be faster, but more complicated architecture. In the case of this program, I think that the times are pretty fast and I don't need to complicate it to get better performance (just jet).

### Edge Cases
Some situations can break the flow. For example, if you turn your head so that there is only one eye visible, the pointer will stop moving (this is on purpose, to avoid some internal invalid maths).

If the lighting is not good, but the algorithm detect something, the mouse can go anywhere. So, expect erratic behaviour if your face and eyes are not fully visible.
