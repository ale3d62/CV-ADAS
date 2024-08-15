import cv2
from flask import Flask, Response, request
import threading
import sys
import signal
from werkzeug.serving import make_server
import socket

class FrameVisualizer:

    def __init__(self, mode, flaskParameters = {'ip': '0.0.0.0', 'port': 5000}):
        self._mode = mode
        if self._mode == "server":
            self._flaskServer = self.FlaskServer(self, flaskParameters)

        self._should_stop = threading.Event()
        signal.signal(signal.SIGINT, self.signal_handler)


    def showFrame(self, frame):
        if self._mode == "server":
            self._flaskServer.updateFrame(frame)
        elif self._mode == "screen":
            cv2.imshow('Frame',frame)
            cv2.waitKey(1)


    def signal_handler(self, sig, frame):
        print("Stopping Flask Server, please, exit the server page")
        #Stop flask server
        self._flaskServer.stop()
        sys.exit(0)


    class FlaskServer:
        
        def __init__(self, parent, parameters):
            self._parent = parent 
            self._ip = parameters["ip"]
            self._port = parameters["port"]

            self._currentFrame = None
            self._server = None

            #Start flask server in new thread
            self._app = Flask(__name__)

            #Show server address
            with self._app.test_request_context():
                ip = None
                if request.host == "localhost":
                    #Get local ip address
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.settimeout(1)
                        # Connect to external dns to get local ip
                        s.connect(("8.8.8.8", 80))
                        ip = s.getsockname()[0]
                    except Exception:
                        ip = '127.0.0.1'
                    finally:
                        s.close()
                else:
                    ip = self._ip

                print("You can visualize the predictions at: ", end="")
                print("http://"+ip+":"+str(self._port))


            self.setupRoutes()
            self._server = make_server(self._ip, self._port, self._app)
            self._server_thread = threading.Thread(target=self._server.serve_forever)
            self._server_thread.start()            
        

        def setupRoutes(self):
            @self._app.route('/')
            def index():
                return Response(self.displayFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        def displayFrame(self):
            while not self._parent._should_stop.is_set():
                if self._currentFrame is not None:
                    ret, buffer = cv2.imencode('.jpg', self._currentFrame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
        def stop(self):
            if self._server:
                self._server.shutdown()
                self._server_thread.join()
                print("Flask Server Successfuly Stopped")
        
        def updateFrame(self, frame):
            self._currentFrame = frame
