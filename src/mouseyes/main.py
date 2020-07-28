from argparse import ArgumentParser
import cv2
import logging as log
import sys

from mouseyes.face_detection import FaceDetectionModel
from mouseyes.head_pose_estimation import HeadPoseEstimationModel
from mouseyes.facial_landmarks_detection import FacialLandmarksModel
from mouseyes.gaze_estimation import GazeEstimationModel
from mouseyes.input_feeder import InputFeeder
from mouseyes.mouse_controller import MouseController


# 1. video => face detection => cropped face img
# 2.1 cropped face => landmark detection   =>  eyes images
# 2.2 cropped face => head pose estimation =>  headpose
# 3. eyes images + headpose => gaze estimation => gaze direction
# 4. gaze direction => move mouse pointer

DEVICE="CPU"
PRECISION="FP32"
FACE_MODEL='models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml'
HEAD_POSE_MODEL=f'models/intel/head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml'
LANDMARKS_MODEL=f'models/intel/landmarks-regression-retail-0009/{PRECISION}/landmarks-regression-retail-0009.xml'
GAZE_MODEL=f'models/intel/gaze-estimation-adas-0002/{PRECISION}/gaze-estimation-adas-0002.xml'
EXTENSION='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
DEFAULT_INPUT="cam"
DEFAULT_CONFIDENCE=0.5
DEFAULT_LOGFILE="mouseyes.log"
DEFAULT_LOGLEVEL="DEBUG"
MAIN_DISPLAY="display"
MOUSE_PRECISION="high"
MOUSE_SPEED="fast"

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
        parser.add_argument("-ml", "--landmarks_model", required=False, type=str, default=LANDMARKS_MODEL,
                            help="Path to an xml file with a trained model for Head Landmarks extraction.")
        parser.add_argument("-mg", "--gaze_model", required=False, type=str, default=GAZE_MODEL,
                            help="Path to an xml file with a trained model for Gaze estimation.")
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
        parser.add_argument("-dev", "--dev", required=False, action="store_true",
                            help="Set options to ease the development.\n"
                            "Same as using -s")
        parser.add_argument("--display_landmarks", required=False, action="store_true",
                            help="Display the landmarks in the output video")
        parser.add_argument("-mp", "--mouse_precision", type=str, default=MOUSE_PRECISION,
                            help="Precision of the mouse pointer. Possible values: high, medium, low")
        parser.add_argument("-ms", "--mouse_speed", type=str, default=MOUSE_SPEED,
                            help="Speed of the mouse pointer. Possible values: fast, medium, slow")
        return parser

    def sanitize_input(self, args):
        if not args.input.lower() == "cam" and not args.input == "0":
            args.input = os.path.abspath(args.input)    #opencv/ffmeg requires full path on my laptop

        #high, medium, low mouse_precision
        mp = {'high', 'medium', 'low'}
        ms = {'fast', 'medium', 'slow'}

        if args.mouse_precision not in mp:
            print('Invalid values for mouse_precision. Please use high, medium or low', file=sys.stderr)
            exit(1)
        if args.mouse_speed not in ms:
            print('Invalid values for mouse_speed. Please use fast, medium or slow', file=sys.stderr)
            exit(1)

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

    def draw_landmarks(self, frame, landmarks, orig_frame=None, face_coords=None):
        
        height, width = frame.shape[0:2]            #https://knowledge.udacity.com/questions/283702
        xmin, ymin = (0, 0)

        if orig_frame is not None and face_coords is not None:
            xmin, ymin, xmax, ymax = face_coords[0]
            frame = orig_frame

        for i in range(0,4,2):
            cv2.circle(frame, (int(landmarks[i]*width)+xmin, int(landmarks[i+1]*height)+ymin),
                                radius=10, color=(255,255,0), thickness=2)
        return frame

    def draw_info(self, frame, info):
        cv2.putText(frame, info, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), thickness=2)
    
    def main(self):

        #handle the command line arguments
        args = self.build_argparser().parse_args()
        self.sanitize_input(args)

        #initialize logging
        log.basicConfig(filename=args.logfile, level=args.loglevel)

        #initialize the models
        face_model = FaceDetectionModel(args.face_model, args.device, args.cpu_extension)
        head_pose_model = HeadPoseEstimationModel(args.head_pose_model, args.device, args.cpu_extension)
        landmark_model = FacialLandmarksModel(args.landmarks_model, args.device, args.cpu_extension)
        gaze_model = GazeEstimationModel(args.gaze_model, args.device, args.cpu_extension)
        mouse = MouseController(args.mouse_precision, args.mouse_speed)

        #open video and process the frames
        ifeed = InputFeeder(args.input)
        for frame in ifeed:
            #get the face coords
            face_coords = self.get_face_coords(face_model, frame)
            
            if args.show_window:
                painted_frame = self.draw_masks(frame, face_coords)
                cv2.imshow(MAIN_DISPLAY, painted_frame)
                #wait for a key
                #if face_coords is None:     #only waitkey (shows the window) if no face coords. Else, overwrite before displaying
                key_pressed = cv2.waitKey(30)
                if key_pressed == 27 or key_pressed == 113: #Esc or q
                    break #exit the for frame in ifeed loop

            if face_coords is None:
                continue

            #crop the face
            cropped_face = face_model.get_cropped_face(frame, face_coords)

            #get head angles
            head_angles = self.get_head_angles(head_pose_model, cropped_face)
            yaw, pitch, roll = head_angles
            #print(f"pitch: {pitch}, roll: {roll}, yaw: {yaw}")
            
            # get cropped eyes:
            eyes, landmarks = self.get_cropped_eyes(landmark_model, cropped_face)
            if eyes.shape[0] < 2:
                continue

            right_eye = eyes[0]
            left_eye = eyes[1]

            if args.show_window:
                if args.display_landmarks:
                    #frame_landmarks = self.draw_landmarks(cropped_face, landmarks)
                    #cv2.imshow('landmarks', frame_landmarks)
                    frame_landmarks = self.draw_landmarks(cropped_face, landmarks, frame, face_coords)
                    cv2.imshow(MAIN_DISPLAY, frame_landmarks)
                #cv2.imshow('right', right_eye)
                #cv2.imshow('left', left_eye)
                #wait for a key
                key_pressed = cv2.waitKey(30)
                if key_pressed == 108: #'l' de Landmark
                    args.display_landmarks = not args.display_landmarks
                if key_pressed == 27 or key_pressed == 113: #Esc or q
                    break #exit the for frame in ifeed loop

            # get the gaze estimation
            gaze_estimation = self.get_gaze_estimation(gaze_model, right_eye, left_eye, head_angles)
            gaze_x, gaze_y, gaze_z = gaze_estimation

            gaze_info = f"Gaze: {gaze_x},{gaze_y}, {gaze_z}"
            #print(gaze_info)

            """
            if args.show_window:
                frame_with_info = self.draw_info(painted_frame, gaze_info)
                cv2.imshow(MAIN_DISPLAY, frame_with_info)
                key_pressed = cv2.waitKey(30)
                if key_pressed == 27 or key_pressed == 113: #Esc or q
                    break #exit the for frame in ifeed loop
            """
            direction = "Move to"
            if gaze_x > 0:
                direction += " Left"
            elif gaze_x < 0:
                direction += " Right"
            else:
                direction += " Stay(Horiz)"

            if gaze_y > 0:
                direction += " Up"
            elif gaze_y < 0:
                direction += " Down"
            else:
                direction += " Stay(Vert)"

            if gaze_z > 0:
                direction += " Diagonal1"
            elif gaze_y < 0:
                direction += " Diagonal2"
            else:
                direction += " Stay(Diagonal)"

            print(direction)

            mouse.move(gaze_x, gaze_y)


    def get_face_coords(self, model, frame):
        image = model.preprocess_input(frame)
        landmarks = model.predict(image)
        w = frame.shape[1]
        h = frame.shape[0]
        return model.preprocess_output(landmarks, w, h)

    def get_head_angles(self, model, image, sync=True):
        """
        Get the head angles (pitch, roll, yaw) from the current *image*, using the *model*.
        If sync is True, run the prediction in sync mode, else async.
        """
        img = model.preprocess_input(image)
        out = model.predict(img, sync)
        #pitch = out['angle_p_fc'][0][0] # pitch = up / down (as in "Yes")
        #roll = out['angle_r_fc'][0][0]  # roll = diagonal head movement (as in "What?")
        #yaw = out['angle_y_fc'][0][0]   # yaw  = head to left / right (as in "No")
        #return pitch, roll, yaw         # Tait-Bryan angles (yaw, pitch or roll)
        return model.preprocess_output(out)

    def get_cropped_eyes(self, model, image, sync=True):
        """
        Get the 2 cropped eyes from original *image* of the cropped face, using *model*.
        If sync is True, run the prediction in sync mode, else async.
        """
        img = model.preprocess_input(image)
        landmarks = model.predict(img, sync)
        #x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = out
        eyes = model.get_cropped_eyes(image, landmarks)
        return eyes, landmarks

    def get_gaze_estimation(self, model, right_eye, left_eye, head_pose, sync=True):
        prep_left, prep_right, prep_hp = model.preprocess_input(left_eye, right_eye, head_pose)
        out = model.predict(prep_left, prep_right, prep_hp, sync)
        return model.preprocess_output(out)

if __name__ == '__main__':
    m = MousEyes()
    m.main()