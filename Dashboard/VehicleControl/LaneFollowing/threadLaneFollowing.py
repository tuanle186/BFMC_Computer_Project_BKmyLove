from CarCommunication.threadwithstop import ThreadWithStop
import cv2, threading
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from client import Client
import utils
from PIDcontroller import PIDController

# FLAGS
PARKING = False
DEBUG = True

# SET OF VERTICES TO FORM ROI
roi_vertices = [[(0, 480), (180 , 350), (500 , 350), (720, 480)]]

# TCP connection
connect_established = False

# PID controller parameters
Kp = 0.5
Ki = 0.0016 #0.001 or 0.008
Kd = 0.1 #2 to 5
setpoint = 0
pid_controller = PIDController(Kp, Ki, Kd, setpoint)
initial_deviation = 0
deviation = initial_deviation


class ThreadLaneFollowing(ThreadWithStop):
    def __init__(self, Ip):
        super(ThreadLaneFollowing, self).__init__()
        self.Ip = Ip
        
    def run(self):
        client = Client(server_ip=self.Ip)
        try:
            client.connect()
            connect_established = True
        except:
            connect_established = False

        if PARKING and connect_established:
            threading.Thread(target=client.client_send, args=((("parking")),)).start()
            sys.exit()

        url = "http://" + self.Ip + ":8000/stream.mjpg"
        cap = cv2.VideoCapture(url)

        while self._running:
            _,src_img = cap.read() 
            if src_img is None:
                break
            src_img = cv2.resize(src_img,(720,480))
            cropped_img = utils.process_image(src_img, roi_vertices)
            # try:

            lines ,(x,y), slope, deviation_angle = utils.lane_tracking(cropped_img)
            
            if slope is not None:
                intercept = 400 - slope*360 
                new_x_point = (350-intercept)/slope
                if new_x_point < 0:
                    new_x_point = 0
                elif new_x_point > 720:
                    new_x_point = 720
                #draw bisector line and vertical line
                cv2.line(src_img,(360,400),(int(new_x_point),300),(33,220,208),1)
                cv2.line(src_img,(360,400),(360,300),(220,33,96),1)
            else:
                cv2.line(src_img,(360,400),(360,300),(220,33,96),1)
        
            if lines is None:
                cv2.imshow("black white image",cropped_img)
                cv2.imshow("Image with lines",src_img)
                continue
            
            deviation = x - 360
            control_signal = pid_controller.update(deviation_angle)
            try:
                # print("control signal:",control_signal)
                control_signal = "{:.1f}".format(control_signal)

                if DEBUG:
                    # print("deviation:",deviation)
                    print("control signal:",control_signal)
                    # print("slope of bisector:",slope)
                    # print("deviation angle:",deviation_angle)

                if connect_established:
                    threading.Thread(target=client.client_send, args=(((control_signal)),)).start()
                    # [right_slope]
                    print("Control signal send:", control_signal)
            except: 
                continue
            for line in lines:
                x1,y1,x2,y2=line
                cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)

            #draw init point and deviation
            image = cv2.circle(src_img, (x,y), radius=1, color=(0, 0, 255), thickness=4)
            image = cv2.circle(src_img, (360,400), radius=1, color=(0, 255, 0), thickness=4)

            #draw ROI
            vertices_array = np.array(roi_vertices, dtype=np.int32)  # Convert to NumPy array
            vertices_array = vertices_array.reshape((-1, 1, 2))
            image = cv2.polylines(src_img, [vertices_array], isClosed=True, color=(0, 0, 255), thickness=2)
            
            cv2.imshow("black white image",cropped_img)
            cv2.imshow("Image with lines",src_img)
            if cv2.waitKey(10) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        super(ThreadLaneFollowing,self).stop()