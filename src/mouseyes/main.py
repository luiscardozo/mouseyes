from argparse import ArgumentParser
import cv2
import logging as log

from mouseyes.face_detection import FaceDetectionModel
from mouseyes.head_pose_estimation import HeadPoseEstimationModel
from mouseyes.input_feeder import InputFeeder


# 1. video => face detection => cropped face img
# 2.1 cropped face => landmark detection   =>  eyes images
# 2.2 cropped face => head pose estimation =>  headpose
# 3. eyes images + headpose => gaze estimation => gaze direction
# 4. gaze direction => move mouse pointer

DEVICE="CPU"
PRECISION="FP32"
FACE_MODEL='models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
HEAD_POSE_MODEL=f'models/intel/head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
DEFAULT_INPUT="cam"
DEFAULT_CONFIDENCE=0.5
DEFAULT_LOGFILE="mouseyes.log"
DEFAULT_LOGLEVEL="DEBUG"

class MousEyes:

    def build_argparser(self):
        """
        Parse command line arguments.

        :return: command line arguments
        """
        parser = ArgumentParser()
        parser.add_argument("-mf", "--face_model", required=False, type=str, default=FACE_MODEL,
                            help="Path to an xml file with a trained model for Face Detection.")
        parser.add_argument("-mh", "--head_pose_model", required=False, type=str, default=HEAD_POSE_MODEL,
                            help="Path to an xml file with a trained model for Head Pose Estimation.")
        parser.add_argument("-i", "--input", required=False, type=str, default=DEFAULT_INPUT,
                            help="Path to image or video file. 'cam' for WebCam")
        parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                            default=EXTENSION,
                            help="MKLDNN (CPU)-targeted custom layers."
                                "Absolute path to a shared library with the"
                                "kernels impl.")
        parser.add_argument("-d", "--device", type=str, default=DEVICE,
                            help="Specify the target device to infer on: "
                                "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                                "will look for a suitable plugin for device "
                                "specified (CPU by default)")
        parser.add_argument("-pt", "--prob_threshold", type=float, default=DEFAULT_CONFIDENCE,
                            help="Probability threshold for detections filtering"
                            "(0.5 by default)")
        parser.add_argument("-o", "--output_video", type=str, default="out.mp4",
                            help="Name of the output video")
        parser.add_argument("-f", "--disable_video_file", required=False, action="store_true",
                            help="Disable the output video file creation")
        parser.add_argument("-s", "--show_window", required=False, action="store_true",
                            help="Shows a Window with the processed output of the image or video")
        parser.add_argument("-k", "--skip_frames", type=int, default=0,
                            help="Skip # of frames on the start of the video.")
        parser.add_argument("-L", "--logfile", type=str, default=DEFAULT_LOGFILE,
                            help="Path to the file to write the log")
        parser.add_argument("-ll", "--loglevel", type=str, default=DEFAULT_LOGLEVEL,
                            help="Level of verbosity log")
        parser.add_argument("--dev", required=False, action="store_true",
                            help="Set options to ease the development.\n"
                            "Same as using -s")
        return parser

    def sanitize_input(self, args):
        if not args.input.lower() == "cam" and not args.input == "0":
            args.input = os.path.abspath(args.input)    #opencv/ffmeg requires full path on my laptop

        if args.dev:
            args.show_window = True
            #and others in the future

    def draw_masks(self, frame, coords):
        '''
        Draw bounding boxes onto the frame.
        '''
        if coords is None:
            return frame
        
        for box in coords:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        return frame

    def main(self):

        #handle the command line arguments
        args = self.build_argparser().parse_args()
        self.sanitize_input(args)

        #initialize logging
        log.basicConfig(filename=args.logfile, level=args.loglevel)

        #initialize the models
        face_model = FaceDetectionModel(args.face_model, args.device, args.cpu_extension)
        head_pose_model = HeadPoseEstimationModel(args.head_pose_model, args.device, args.cpu_extension)
        #open video and process the frames
        ifeed = InputFeeder(args.input)
        for frame in ifeed:
            #get the cropped face
            image = face_model.preprocess_input(frame)
            out = face_model.predict(image)
            w = frame.shape[1]
            h = frame.shape[0]
            proc_out = face_model.preprocess_output(out, w, h)
            
            if args.show_window:
                painted_frame = self.draw_masks(frame, proc_out)
                cv2.imshow('display', painted_frame)
                #wait for a key
                key_pressed = cv2.waitKey(30)
                if key_pressed == 27 or key_pressed == 113: #Esc or q
                    break #exit the for frame in ifeed loop

            if proc_out is None:
                continue

            cropped_img = face_model.get_cropped_face(frame, proc_out, True)
            print(type(cropped_img))
            print(cropped_img.shape)
            pitch, roll, yaw = self.get_head_angles(head_pose_model, cropped_img)
            print(f"pitch: {pitch}, roll: {roll}, yaw: {yaw}")
            
            # pass cropped_img to next 

    def get_head_angles(self, model, image):
        img = model.preprocess_input(image)
        out = model.predict(img, True)
        pitch = out['angle_p_fc'][0][0]
        roll = out['angle_r_fc'][0][0]
        yaw = out['angle_y_fc'][0][0]
        return pitch, roll, yaw

if __name__ == '__main__':
    m = MousEyes()
    m.main()