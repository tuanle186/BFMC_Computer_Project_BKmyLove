import cv2, threading
import numpy as np
import matplotlib.pyplot as plt
from client import Client
# from server import Server

def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255) 
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def middle_lane_point(lines):
    x_right_list = []
    x_left_list = []

    x_right = 720
    x_left = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if max([y1,y2])<=300:
            continue
        # fit = np.polyfit((x1,x2), (y1,y2), 1)
        if (x2-x1) == 0:
            return (360,350)
        
        slope = (y2-y1)/(x2-x1)
        ave_x = (x1+x2)/2

        if slope < 0: 
            x_left_list.append(ave_x)
        else:
            x_right_list.append(ave_x)
       
    if len(x_left_list) == 0:
        x_left = -200
    else:
        x_left = np.average(x_left_list)
    if len(x_right_list) == 0:
        x_right = 920
    else:
        x_right = np.average(x_right_list)

    x = int((x_right+x_left)/2)
    return (x, 350)

def lane_tracking(edges):
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=30, # Min number of votes for valid line
                minLineLength=10, # Min allowed length of line
                maxLineGap=4# Max allowed gap between line for joining them
                )
    if lines is None:
        return [],(0,0)
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        lines_list.append([x1,y1,x2,y2])
    (x,y) = middle_lane_point(lines)
    return lines_list, (x,y)

def process_image(image, roi_vertices):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.blur(gray_img, (5,5), cv2.BORDER_DEFAULT) 
    edges = cv2.Canny(blur_img,190,230,None, 3)
    cropped_img = roi(edges, np.array([roi_vertices],np.int32))
    return cropped_img


ENABLE_VID = True
roi_vertices = [[(0, 480), (200 , 300), (520 , 300), (720, 480)]]
connect_established = False

if __name__ == "__main__":
    # thread = threading.Thread(target=sendData)
    # thread.start()
    # server = Server()
    client = Client()
    try:
        client.connect()
        connect_established = True
    except:
        connect_established = False
    
    try:
        if ENABLE_VID:
            # cap = cv2.VideoCapture("duong_cong_2.mp4")
            url = "http://192.168.0.101:8000/stream.mjpg"
            cap = cv2.VideoCapture(url)
            pre_dev_x = 0
            while True:
                _,src_img = cap.read() 
                src_img = cv2.resize(src_img,(720,480))
                cropped_img = process_image(src_img, roi_vertices)
                image = src_img
                # try:

                lines ,(x,y) = lane_tracking(cropped_img)
                if lines is None:
                    cv2.imshow("black white image",cropped_img)
                    cv2.imshow("Image with lines",image)
                    continue

                dev_x = x - 360
                if abs(dev_x-pre_dev_x) > 15:
                    print("x_deviation: ",dev_x)
                    if connect_established:
                        threading.Thread(target=client.client_send, args=(str(dev_x),)).start()
                    pre_dev_x = dev_x

                for line in lines:
                    x1,y1,x2,y2=line
                    # Draw the lines joing the points
                    # On the original image
                    cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)
                deviation = x
                image = cv2.circle(src_img, (x,y), radius=1, color=(0, 0, 255), thickness=4)
                image = cv2.circle(src_img, (360,350), radius=1, color=(0, 255, 0), thickness=4)
                # image = cv2.line(src_img, (0,300) , (720,300), (0,0,255), 2)
                vertices_array = np.array(roi_vertices, dtype=np.int32)  # Convert to NumPy array
                vertices_array = vertices_array.reshape((-1, 1, 2))
                image = cv2.polylines(src_img, [vertices_array], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.imshow("black white image",cropped_img)
                cv2.imshow("Image with lines",image)
                if cv2.waitKey(10) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        else:
            src_img = cv2.imread('img_test/left1.png')
            pre_x = 0
            src_img = cv2.resize(src_img,(720,480))
            cropped_img = process_image(src_img, roi_vertices)
            image = src_img
            # try:
            lines ,(x,y) = lane_tracking(cropped_img)
            if abs(x-pre_x) < 20:
                x = pre_x
            pre_x = x
            for line in lines:
                x1,y1,x2,y2=line
                # Draw the lines joing the points
                # On the original image
                cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)
            image = cv2.circle(src_img, (x,y), radius=1, color=(0, 0, 255), thickness=4)
            image = cv2.circle(src_img, (360,350), radius=1, color=(0, 255, 0), thickness=4)
            # image = cv2.line(src_img, (0,300) , (720,300), (255,0,255), 4)
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
            plt.imshow(image)
            plt.title('img')

            plt.show()
    except KeyboardInterrupt:
        print("error: KeyboardInterrupt")