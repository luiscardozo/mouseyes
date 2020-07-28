'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
import os
from numpy import ndarray
import logging as log

class InputFeeder:
    def __init__(self, input_file, as_file=False):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self._as_file = as_file
        self.input_file = input_file
        self.input_type=self.video_or_pic(input_file)

        if self.input_type=='video' or self.input_type=='image':
            if not os.path.exists(input_file):
                raise FileNotFoundError(os.path.abspath(input_file))
        if self.input_type == 'video' or self.input_type == 'cam':
            in_video = 0 if self.input_type == 'cam' else input_file
            self._cap = cv2.VideoCapture(in_video)
            self._cap.open(in_video)
            self._width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) #OpenCV 3+
            print(f"Total frames: {self._total_frames}")
        self._frame_nr = 0

    def __iter__(self):
        "Turn InputFeeder into an Iterable"
        return self

    def __next__(self):
        "Get the next frame of the video or the next image"
        if self.input_type == 'image':
            self._frame_nr += 1
            if self._frame_nr > 1:
                raise StopIteration
            else:
                return cv2.imread(self.input_file)
        else:
            if not self._cap.isOpened:
                raise StopIteration
            else:
                self._frame_nr += 1
                #print(f"Frame {self._frame_nr} of {self._total_frames}")
                flag, raw_frame = self._cap.read()
                if not flag:
                    self._close_cap()
                    raise StopIteration

                if self._as_file:
                    return self._save_image(raw_frame)
                else:
                    return raw_frame
            
    def __len__(self):
        return 1 if self.input_type == 'image' else self._total_frames

    def _close_cap(self):
        self._cap.release()
        cv2.destroyAllWindows()

    def _save_image(self, frame):
        img_path = f'tmp/img{self._frame_nr}.jpg'
        cv2.imwrite(img_path, frame)
        return img_path

    def video_or_pic(self, input_path):
        """
        Check if input file is a supported image or movie.
        If not, notify the user with an exception.
        Copied from my work in "person counter"
        """
        # based on https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
        pics = ["jpg", "jpeg", "png", "gif", "bmp", "jpe", "jp2", "tiff", "tif"]
        movs = ["avi", "mpg", "mp4", "mkv", "ogv"]

        input_type = None
        
        ext = os.path.splitext(input_path)[1][1:]
        if ext in pics:
            input_type = 'image'
        elif ext in movs:
            input_type = 'video'
        else:
            if input_path.lower() == "cam" or input_path == "0":
                input_type = 'cam'
            else:
                log.warning("Input file format not supported")
        
        return input_type

if __name__ == '__main__':
    TEST_IMAGE='../../tests/resources/center.jpg'
    ifeed = InputFeeder('image', TEST_IMAGE)
    ifeed.load_data()
    print(ifeed.cap)
    print(type(ifeed.cap))
    print(ifeed.cap.shape)