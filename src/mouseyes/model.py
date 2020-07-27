'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import logging as log
import sys
import cv2
import numpy as np

from openvino.inference_engine import IECore, IENetwork
from mouseyes.mouseyes_exceptions import UnsupportedLayersException, ModelXmlFileNotFoundException, ModelBinFileNotFoundException

class ModelBase:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, transpose=(2,0,1)):
        '''
        Initialize the OpenVINO Engine
        '''
        #Get the model structure and weights
        self.model_structure, self.model_weights = self.check_model(model_path)
        
        self.device = device
        self.extensions = extensions
        self.transpose_form = transpose
        self.load_model()

    def load_model(self):
        '''
        Loads the model and the required plugins.
        '''

        self.core = IECore()

        ### Add any necessary extensions ###
        if self.extensions and "CPU" in self.device:
            self.core.add_extension(self.extensions, self.device)

        # read the network structure and create an IENetwork
        self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        # Or in 2020.2+ =>
        #self.network = self.core.read_network(model=model_structure, weights=model_weights)

        ### Check for supported layers ###
        #if self.unsupported_layers(self.network, self.device):
        #    raise UnsupportedLayersException()

        # Get input and output names
        self.input_name=next(iter(self.network.inputs))    #only for models with 1 input. Review later (eg.: Resnet have >1)
        self.output_name=next(iter(self.network.outputs))
        
        try:
            self.exec_net = self.core.load_network(network=self.network, device_name=self.device) #num_requests=1
        except Exception as e:
            if "unsupported layer" in str(e):
                # OpenVINO throws a RuntimeException on unsupported layer,
                # not an specific type of exception
                print("Cannot run the model, unsupported layer: ", e, file=sys.stderr)
                print("You can try to pass a CPU Extension with the argument --cpu_extension", file=sys.stderr)
            else:
                print(e, file=sys.stderr)
            raise UnsupportedLayersException()


    def predict(self, image, sync=False, request_id=0):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_blob = {self.input_name: image}

        if sync:
            return self.infer_sync(input_blob) #return output blob
        else:
            self.infer_async(input_blob, request_id)
            if self.wait(request_id) == 0:
                return self.get_output(request_id)
    
    def infer_sync(self, input_blob):
        return self.exec_net.infer(input_blob)

    def infer_async(self, input_blob, request_id):
        self.exec_net.start_async(request_id=request_id, inputs=input_blob)

    def check_model(self, model_path):
        #To work even if you sent with .xml or not
        filename, _ = os.path.splitext(model_path)
        model_bin = filename+'.bin'
        model_xml = filename+'.xml'

        if not os.path.exists(model_xml):
            err_msg = "Model XML File does not exist in path " + os.path.abspath(model_xml)
            log.error(err_msg)
            raise ModelXmlFileNotFoundException(err_msg)
        
        if not os.path.exists(model_bin):
            err_msg = "Model XML File does not exist on path " + model_bin
            log.error(err_msg)
            raise ModelBinFileNotFoundException(err_msg)

        return model_xml, model_bin    
    
    def preprocess_input(self, image, required_size=None):
        """
        Preprocess the frame according to the model needs.
        """
        if required_size is None:
            input_shape = self.get_input_shape()
            required_size = (input_shape[3], input_shape[2])
            
        frame = cv2.resize(image, required_size)    #ToDo: mover a input_feeder?
        #cv2.cvtColor if not BGR
        frame = frame.transpose(self.transpose_form)    #depends on the model. Initialized with the class
        frame = frame.reshape(1, *frame.shape)          #depends on the model
        return frame

    def preprocess_output(self, outputs, vid_width, vid_height, label=1.0, min_threshold=0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        if outputs.shape == (1,1,200,7):
            coords = []
            for box in outputs[0][0]:
                if box[1] != label:
                    continue
                if box[2] < min_threshold:
                    continue

                xmin = int(box[3] * vid_width)
                ymin = int(box[4] * vid_height)
                xmax = int(box[5] * vid_width)
                ymax = int(box[6] * vid_height)
                coords.append([xmin, ymin, xmax, ymax])
            out = np.array(coords)
            if out.shape == (0, ):
                return None
            else:
                return np.array(coords)
        else:
            raise NotImplementedError   #Implement in each subclass
    
        
    def wait(self, request_id=0, timeout_ms=-1):
        """Waits for an async request to be complete."""
        return self.exec_net.requests[request_id].wait(timeout_ms)

    def get_output(self, request_id=0, output_name=None):
        """Return the output of a finished async request"""
        if output_name is None:
            output_name = self.output_name
        
        return self.exec_net.requests[request_id].outputs[output_name]

    def get_input_shape(self):
        return self.network.inputs[self.input_name].shape

    def get_output_shape(self):
        return self.network.outputs[self.output_name].shape

    def unsupported_layers(self, net, device):
        """
        Checks for unsupported layers on selected device.
        Returns a set of unsupported layers (empty if there are none)
        """
        supported_layers = self.core.query_network(network=net, device_name=device)

        unsupported = set()
        for layer, obj in net.layers.items():
            if not layer in supported_layers and obj.type != 'Input':
                log.debug(f"Unsupported layer: {layer}, Type: {obj.type}")
                unsupported.add(obj.type)

        if len(unsupported) != 0:
            print("There are unsupported layers in the current model.\n"
                  "Try to load a CPU extension to solve this problem.\n"
                  "Layer types: " + str(unsupported) + "\n"
                  "Cannot continue. Exiting.", file=sys.stderr)
            return True

        return False