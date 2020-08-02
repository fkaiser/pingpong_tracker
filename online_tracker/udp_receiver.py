import socket
import cv2
import time
import sys
import os
import argparse

def udp_opencv(ip='127.0.0.1', port=5000, store_video=False, store_image=False):
    cap = cv2.VideoCapture('udp://'+ ip + ':' + str(port) + '?overrun_nonfatal=1&fifo_size=50000000')
    if not cap.isOpened():
        print('VideoCapture not opened')
        exit(-1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate_store = 20
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    if store_video:
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate_store, (frame_width,frame_height))
    base_name = 'testimages/'
    counter = 0
    counter_freq = 0
    print('Starting')
    start = time.time()
    while True:
        ret, frame = cap.read()
        counter_freq += 1
        if not ret:
            print('frame empty')
        hough_img = track_ball_hough(frame)
        cv2.imshow('image', hough_img)

        if store_video:
            out.write(hough_img)
        if store_image:
            store_name = base_name + str(counter) + '.png'
            cv2.imwrite(store_name, hough_img) 
        if counter < 1000:
            counter += 1
        else:
            counter = 1
        if cv2.waitKey(1)&0XFF == ord('q'):
            diff_time = time.time() - start
            freq  = counter_freq / diff_time
            print('Frequency is: {}'.format(freq))
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def convert_to_grayscale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray


def track_ball_hough(image):
        return image
        img = cv2.medianBlur(convert_to_grayscale(image), 5)
        cimg = image
        circles = cv2.HoughCircles(img ,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=10,maxRadius=50)
        if False and not circles is None:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        return cimg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip',
                        help='IP of receiver')
    parser.add_argument('--port', default=5000, type=int,
                        help='Port to send stream to.')
    parser.add_argument('--store_video', action='store_true',
                        help='Option to show each frame with particles')
    parser.add_argument('--store_images', action='store_true',
                        help='Option to show each frame with particles')                  
    args = parser.parse_args()
    udp_opencv(ip=args.ip, port=args.port, store_video=args.store_video, store_image=args.store_images)


def udp_raw():
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    address = ("192.168.1.13", 5001)
    sock.bind(address)
    while True:
        data, addr = sock.recvfrom(1024)
        print(addr)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)