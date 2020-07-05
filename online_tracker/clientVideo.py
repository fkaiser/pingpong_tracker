import socket
import time
import picamera
import os
import sys
import argparse

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
def stream_via_udp(ip, port):
    scan_host()
    while True:

        receiver_ready = scan_host(host=ip, port=port)
        if receiver_read == 0:
            break
        else:
            print('Waiting until UDP receiver on ip {} and port {} is ready'.format(ip, port))
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


def scan_host(host, port, r_code = 1) : 
    try : 
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        code = s.connect_ex((host, port))
        if code == 0 : 
            r_code = code
        s.close()
    except Exception, e : 
        pass
    return r_code


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
