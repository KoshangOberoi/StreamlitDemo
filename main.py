import cv2
import argparse
import asyncio
import ssl
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import uuid
from aiohttp import web
from av import VideoFrame
import logging
import json
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        mixer.init()
        sound = mixer.Sound('alarm.wav')


        face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
        leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
        reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


        lbl=['Close','Open']

        model = load_model('models/cnncat2.h5')
        path = os.getcwd()
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        count=0
        score=0
        thicc=2
        rpred=[99]
        lpred=[99]

        def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
            """Return the 3D points present as 2D for making annotation box"""
            point_3d = []
            dist_coeffs = np.zeros((4,1))
            rear_size = val[0]
            rear_depth = val[1]
            point_3d.append((-rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, -rear_size, rear_depth))
            
            front_size = val[2]
            front_depth = val[3]
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d.append((-front_size, front_size, front_depth))
            point_3d.append((front_size, front_size, front_depth))
            point_3d.append((front_size, -front_size, front_depth))
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
            
            # Map to 2d img points
            (point_2d, _) = cv2.projectPoints(point_3d,
                                            rotation_vector,
                                            translation_vector,
                                            camera_matrix,
                                            dist_coeffs)
            point_2d = np.int32(point_2d.reshape(-1, 2))
            return point_2d

        def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                                rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                                color=(255, 255, 0), line_width=2):
            """
            Draw a 3D anotation box on the face for head pose estimation

            Parameters
            ----------
            img : np.unit8
                Original Image.
            rotation_vector : Array of float64
                Rotation Vector obtained from cv2.solvePnP
            translation_vector : Array of float64
                Translation Vector obtained from cv2.solvePnP
            camera_matrix : Array of float64
                The camera matrix
            rear_size : int, optional
                Size of rear box. The default is 300.
            rear_depth : int, optional
                The default is 0.
            front_size : int, optional
                Size of front box. The default is 500.
            front_depth : int, optional
                Front depth. The default is 400.
            color : tuple, optional
                The color with which to draw annotation box. The default is (255, 255, 0).
            line_width : int, optional
                line width of lines drawn. The default is 2.

            Returns
            -------
            None.

            """
            
            rear_size = 1
            rear_depth = 0
            front_size = img.shape[1]
            front_depth = front_size*2
            val = [rear_size, rear_depth, front_size, front_depth]
            point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
            # # Draw all the lines
            cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(img, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(img, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(img, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)
            
            
        def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
            """
            Get the points to estimate head pose sideways    

            Parameters
            ----------
            img : np.unit8
                Original Image.
            rotation_vector : Array of float64
                Rotation Vector obtained from cv2.solvePnP
            translation_vector : Array of float64
                Translation Vector obtained from cv2.solvePnP
            camera_matrix : Array of float64
                The camera matrix

            Returns
            -------
            (x, y) : tuple
                Coordinates of line to estimate head pose

            """
            rear_size = 1
            rear_depth = 0
            front_size = img.shape[1]
            front_depth = front_size*2
            val = [rear_size, rear_depth, front_size, front_depth]
            point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
            y = (point_2d[5] + point_2d[8])//2
            x = point_2d[2]
            
            return (x, y)

        face_model = get_face_detector()
        landmark_model = get_landmark_model()

        outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
        d_outer = [0]*5
        inner_points = [[61, 67], [62, 66], [63, 65]]
        d_inner = [0]*3

        ret, frame = cap.read()
        size = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )

        while(True):
            ret, frame = cap.read()
            rects = find_faces(frame, face_model)
            for rect in rects:
                shape = detect_marks(frame, landmark_model, rect)
                draw_marks(frame, shape)
                cv2.putText(frame, 'Press r to record Mouth distances', (30, 30), font,
                            1, (0, 255, 255), 2)
                cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                for i in range(100):
                    for i, (p1, p2) in enumerate(outer_points):
                        d_outer[i] += shape[p2][1] - shape[p1][1]
                    for i, (p1, p2) in enumerate(inner_points):
                        d_inner[i] += shape[p2][1] - shape[p1][1]
                break
        cv2.destroyAllWindows()
        d_outer[:] = [x / 100 for x in d_outer]
        d_inner[:] = [x / 100 for x in d_inner]

        while(True):
            ret, frame = cap.read()
            height,width = frame.shape[:2] 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = find_faces(frame, face_model)
            # faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

            # for (x,y,w,h) in faces:
            #     cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            rects = find_faces(frame, face_model)
            for rect in rects:
                shape = detect_marks(frame, landmark_model, rect)
                cnt_outer = 0
                cnt_inner = 0
                draw_marks(frame, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        cnt_outer += 1 
                for i, (p1, p2) in enumerate(inner_points):
                    if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                        cnt_inner += 1
                if cnt_outer > 3 and cnt_inner > 2:
                    print('Mouth open')
                    cv2.putText(frame, 'Mouth open', (30, 30), font,
                            1, (0, 255, 255), 2)
            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                count=count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = np.argmax(model.predict(r_eye),axis=-1)
                if(rpred[0]==1):
                    lbl='Open' 
                if(rpred[0]==0):
                    lbl='Closed'
                break

            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                count=count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = np.argmax(model.predict(l_eye),axis=-1)
                if(lpred[0]==1):
                    lbl='Open'   
                if(lpred[0]==0):
                    lbl='Closed'
                break

            if(rpred[0]==0 and lpred[0]==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
            else:
                score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
                
            if(score<0):
                score=0   
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            if(score>5):
                #person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(path,'image.jpg'),frame)
                try:
                    sound.play()
                    
                except:  # isplaying = False
                    pass
                if(thicc<16):
                    thicc= thicc+2
                else:
                    thicc=thicc-2
                    if(thicc<2):
                        thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
            faces = find_faces(frame, face_model)
            for face in faces:
                marks = detect_marks(frame, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],     # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                        ], dtype="double")
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                    
                    
                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose
                    
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    
                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                    
                    
                    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    x1, x2 = head_pose_points (frame, rotation_vector, translation_vector, camera_matrix)

                    cv2.line(frame, p1, p2, (0, 255, 255), 2)
                    cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)
                    # for (x, y) in marks:
                    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                    try:
                        m = (p2[1] - p1[1])/(p2[0] - p1[0])
                        ang1 = int(math.degrees(math.atan(m)))
                    except:
                        ang1 = 90
                        
                    try:
                        m = (x2[1] - x1[1])/(x2[0] - x1[0])
                        ang2 = int(math.degrees(math.atan(-1/m)))
                    except:
                        ang2 = 90
                        
                        # print('div by zero error')
                    if ang1 >= 48:
                        print('Head down')
                        cv2.putText(frame, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                    elif ang1 <= -48:
                        print('Head up')
                        cv2.putText(frame, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
                    
                    if ang2 >= 48:
                        print('Head right')
                        cv2.putText(frame, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                    elif ang2 <= -48:
                        print('Head left')
                        cv2.putText(frame, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                    
                    cv2.putText(frame, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                    cv2.putText(frame, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# async def on_shutdown(app):
    # close peer connections
    # coros = [pc.close() for pc in pcs]
    # await asyncio.gather(*coros)
    # pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    # app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
