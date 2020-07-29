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


## Benchmarks
### Testing with other hardwares
As I have only a 3rd gen i7 (OpenVINO officially supports only 6th to 10th gen Intel Processors), I had to use CPU extensions in order to be able to run the system in my CPU.
Trying to run the software in GPU, gives me an error: `clGetPlatformIDs error -1001`
Trying to run in NCS2 (MYRIAD) gives me `unsupported layer type "Quantize"`
Trying to run in `HETERO:MYRIAD,CPU` or `HETERO:GPU,CPU`, gives me an error because I need to apply the CPU extension but HETERO cannot accept the extension.

So, for now, I can only test in CPU with extensions, until I get a newer hardware.

*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
Some situations can break the flow. For example, if you turn your head so that there is only one eye visible, the pointer will stop moving (this is on purpose, to avoid some internal invalid maths).

There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
