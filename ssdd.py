from __future__ import division
import time
from collections import Counter
import argparse
import os
import time
import pickle as pkl
import torch
from torch.autograd import Variable
import cv2
import glob
from util.parser import load_classes
from util.model import Darknet
from util.image_processor import prep_image
from util.utils import non_max_suppression



glob.minutes_=0.5
glob.time_=glob.minutes_ * 60
glob.start_ = time.time()


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--config", dest = 'configfile', help =  "Config file", default = "config/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "weights/yolov3.weights", type = str)
    parser.add_argument("--outputs", dest = 'outputs', help = "Image / Directory to store detections", default = "outputs", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", type = str)
    parser.add_argument("--cam", dest = "camera", help = "use camera to make detections", default=False, action="store_true")

    return parser.parse_args()

args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.configfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    print(1234567)
    model.cuda()


#Set the model in evaluation mode
model.eval()
obj_counter = {}

#Make output dir if it's not exist
if not os.path.exists(args.outputs):
    os.makedirs(args.outputs)


with open("mail.html", "r", encoding='utf-8') as f:
    mailFormat= f.read()







def mail(maximum_):
    import pandas as pd
    df=pd.DataFrame(glob.list_)
    df.to_csv("log.csv")

    import smtplib
    import os
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders
    from pathlib import Path

    FROM = "socialdistancing.drone@gmail.com"
    mail="ab5531@bennett.edu.in"
    TO=mail
    msg = MIMEMultipart()
    msg['Subject'] = 'Alert: Generated Report'
    msg['To'] = TO
    msg['From'] = FROM
    msg.preamble = """Your mail reader does not support the report format."""
    name=mail.split('@')[0]
    text = MIMEText(str(mailFormat).format(name,maximum_),'html')
    part = MIMEBase('application', "octet-stream")

    #
    img_data = open('image.png', 'rb').read()
    image_data = MIMEImage(img_data, name=str("Image")+".png")
    msg.attach(image_data)

    with open("log.csv", 'rb') as file:
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',
                    'attachment; filename="{}"'.format(Path("log.csv").name))
    msg.attach(part)
    msg.attach(text)
    TO = [mail]

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
    except:
         server = smtplib.SMTP('smtp.gmail.com', 465)
    server.starttls()
    server.login("socialdistancing.drone@gmail.com", "Abhi@2699")
    server.sendmail(FROM, TO, msg.as_string())
    server.quit()

def timer(maximum_):
    import glob
    duration=time.time() - glob.start_
    if duration>glob.time_:
            mail(maximum_)
            glob.start_=time.time()
            glob.maxs_=-1000
            glob.list_={"Time":[],"Number of People":[],"Image":[]}



def write(x, results):
    import glob
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results

    cls = int(x[-1])
    color = colors[cls%100]
    label = "{0}: {1}".format(classes[cls],str(obj_counter[cls]))
    if((int(obj_counter[cls]))>5  and (classes[cls]=='person')):

        glob.list_['Time'].append(time.time())
        glob.list_['Number of People'].append(int(obj_counter[cls]))
        glob.list_['Image'].append(img)

        if glob.maxs_<int(obj_counter[cls]):
            from PIL import Image
            im = Image.fromarray(img)
            im.save("image.png")
            glob.maxs_=int(obj_counter[cls])
        timer(glob.maxs_)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase
if args.videofile:
    videofile = args.videofile #or path to the video file.
    video_path = os.path.join(args.outputs,videofile.split('/')[0].split('\\')[-1])
    print("Path:",videofile )
    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'

elif args.camera:
    cap = cv2.VideoCapture(0)
    video_path = os.path.join(args.outputs,'cam_output.mp4')
else:
    raise "there is not video or camera option choosen, please chose one option to start work on it"




frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
frames = 0
start = time.time()

while True:
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img, volatile = True))
        output = non_max_suppression(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue


        obj_counter = output[:,-1]
        obj_counter = Counter(obj_counter.numpy().astype(int).tolist())


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("colors/pallete", "rb"))
        clss = {}
        list(map(lambda x: write(x, frame), output))
        out.write(frame)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or not cap.isOpened() or not cap.isOpened():
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
