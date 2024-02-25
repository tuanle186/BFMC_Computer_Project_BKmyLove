import cv2, threading
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from client import Client
# from server import Server
import utils
from PIDcontroller import PIDController

# Raspberry Pi's IP address
with open("../../setup/PairingData.json", "r") as file:
    data = json.load(file)
Ip = data["ip"]

# PARKING FLAG
PARKING = False

# VID OR IMAGE
ENABLE_VID = True
DEBUG = True

# SET OF VERTICES TO FORM ROI
roi_vertices = [[(0, 480), (180 , 300), (540 , 300), (720, 480)]]

# TCP connection
connect_established = False

# PID controller parameters
Kp = 0.5
Ki = 0 #0.001 or 0.008
Kd = 0 #2 to 5
setpoint = 0
pid_controller = PIDController(Kp, Ki, Kd, setpoint)
initial_deviation = 0
deviation = initial_deviation

if __name__ == "__main__":
    # thread = threading.Thread(target=sendData)
    # thread.start()
    # server = Server()
    client = Client(server_ip=Ip)
    try:
        client.connect()
        connect_established = True
    except:
        connect_established = False

    if PARKING and connect_established:
        threading.Thread(target=client.client_send, args=((("parking")),)).start()
        sys.exit()

    try:
        if ENABLE_VID:
            # cap = cv2.VideoCapture("very_ok.mp4")
            url = "http://" + Ip + ":8000/stream.mjpg"
            cap = cv2.VideoCapture(url)
            while True:
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

        else:
            src_img = cv2.imread('img_test/left8.png')
            # src_img = cv2.imread('img_real/img1.png')
            src_img = cv2.resize(src_img,(720,480))
            cropped_img = utils.process_image(src_img, roi_vertices)
            # try:
            lines ,(x,y), slope, deviation_angle = utils.lane_tracking(cropped_img)
            intercept = 400 - slope*360 
            new_x_point = (350-intercept)/slope
            
            #draw bisector line and vertical line
            cv2.line(src_img,(360,400),(int(new_x_point),300),(33,220,208),1)
            cv2.line(src_img,(360,400),(360,300),(220,33,96),1)

            deviation = x - 360
            control_signal = pid_controller.update(deviation)
            
            if DEBUG:
                print("deviation:",deviation)
                print("control signal:",control_signal)
                print("slope of bisector:",slope)
                print("deviation angle:",deviation_angle)

            for line in lines:
                x1,y1,x2,y2=line
                cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)

            image = cv2.circle(src_img, (x,y), radius=1, color=(0, 0, 255), thickness=4)
            image = cv2.circle(src_img, (360,400), radius=1, color=(0, 255, 0), thickness=4)
            
            vertices_array = np.array(roi_vertices, dtype=np.int32)  # Convert to NumPy array
            vertices_array = vertices_array.reshape((-1, 1, 2))
            image = cv2.polylines(src_img, [vertices_array], isClosed=True, color=(0, 0, 255), thickness=2)
            plt.figure(figsize=(15, 5))

            # Plot the first image in the first subplot
            plt.subplot(1, 2, 1)
            plt.imshow(cropped_img)
            plt.title('edges')

            # Plot the second image in the second subplot
            plt.subplot(1, 2, 2)
            plt.imshow(src_img)
            plt.title('img')

            plt.show()
    except KeyboardInterrupt:
        print("error: KeyboardInterrupt")