import socket
import time
import picamera
import os
import sys
import argparse

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
def stream_via_udp(ip, port):
    client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    client_socket.connect((ip, port))

    # Make a file-like object out of the connection
    connection = client_socket.makefile('wb')
    try:
        camera = picamera.PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 80
        # Start a preview and let the camera warm up for 2 seconds
        camera.start_preview()
        time.sleep(2)
        # Start recording, sending the output to the connection for 60
        # seconds, then stop
        camera.start_recording(connection, format='h264')
        camera.wait_recording(36000)
        camera.stop_recording()
    finally:
        connection.close()
        client_socket.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip',
                        help='IP of receiver')
    parser.add_argument('--port', default=5000, type=int,
                        help='Port to send stream to.')
    args = parser.parse_args()
    stream_via_udp(ip=args.ip, port=args.port)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)