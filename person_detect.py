
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_xml=model_name+'.xml'
        self.model_bin=model_name+'.bin'
        self.device=device
        self.threshold=threshold
        '''
        Create an instance of IECore in order to perform model loading and inference
        '''
        self.ie_core = IECore()

        try:
            '''
            When using IENetwork() as in the course material (Lesson 4 - load_to_IE() method in Inference Request 
            exercise), it gives a 'Deprecated warning' message. So auto-suggested load_network() is used here.            
            '''
            self.model=self.ie_core.read_network(self.model_xml, self.model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
    
    def convert_color(self,color_string):
        '''
        Get the BGR value of the desired bounding box color.
        Defaults to Blue if an invalid color is given.
        '''
        colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
        out_color = colors.get(color_string)
        if out_color:
            return out_color
        else:            
            return colors['BLUE']

    def load_model(self):
        self.exec_net = self.ie_core.load_network(network=self.model, device_name=self.device, num_requests=1)        
    
    def predict(self, image, color):    
        '''
        Pre-process the input image to perform resize, transform and reshape as required
        '''
        p_frame = self.preprocess_input(image)
        
        '''
        Uncomment the below lines and comment out the async inference part
        for performing synchronous inference        
        '''
        # outputs = self.exec_net.infer({self.input_name: p_frame})
        # coords, image = self.preprocess_outputs(outputs[self.output_name], image, color, self.threshold)
        
        '''
        Perform async inference
        '''        
        self.exec_net.start_async(request_id=0, inputs={self.input_name:p_frame})
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break            
        outputs = self.exec_net.requests[0].outputs[self.output_name]
        
        '''        
        Pre-process the outputs to extract and draw the bounding boxes
        '''
        coords, image = self.preprocess_outputs(outputs, image, color, self.threshold)
        return coords, image
    
    def draw_outputs(self, xmin, ymin, xmax, ymax, image, color, thickness):   
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        cv2.rectangle(image, start_point, end_point, self.convert_color(color), thickness)

    def preprocess_outputs(self, outputs, image, color, threshold):
        coords = []
        box_thickness = 2
        
        '''        
        #### output.shape: 1x1xNx7 ####
        [image_id, label, conf, x_min, y_min, x_max, y_max]
        image_id - ID of the image in the batch
        label - predicted class ID
        conf - confidence for the predicted class
        (x_min, y_min) - coordinates of the top left bounding box corner
        (x_max, y_max) - coordinates of the bottom right bounding box corner
        '''
        for bounding_box in outputs[0][0]: 
            confidence_threshold = bounding_box[2]
            if confidence_threshold >= threshold:
                xmin = int(bounding_box[3] * self.w)
                ymin = int(bounding_box[4] * self.h)
                xmax = int(bounding_box[5] * self.w)
                ymax = int(bounding_box[6] * self.h)
                self.draw_outputs(xmin, ymin, xmax, ymax, image, color, box_thickness)
                coords.append((xmin, ymin, xmax, ymax))
        return coords, image

    def preprocess_input(self, image):
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame
    
    # Height and width are required when we draw the bounding boxes.
    # Set them once we are able to determine it from the input feed
    def set_width_height(self, width, height):
        self.w = width
        self.h = height


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    color=args.color
    if not color:
        color = "BLUE"

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:    
        '''
        Set the height and width before we require them to 
        draw bounding boxes
        '''        
        pd.set_width_height(initial_w, initial_h)
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame, color)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--color', default="BLUE")
    
    args=parser.parse_args()

    main(args)
    
    '''
    Note:
    Solutions to Course Exercises used in this project - 
        --- preprocess_inputs, 
        ----preprocess_outputs, 
        ----draw_boxes, 
        ----convert_color
        ----load_to_IE
        ----sync_inference
        ----async_inference
    
    Credits:
    1) https://knowledge.udacity.com/questions/238022
    2) https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Inference-Engine-Python-API-Error/td-p/1148871
    '''