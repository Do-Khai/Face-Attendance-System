import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os
import glob
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

IMG_PATH = './data/test_images/'
DATA_PATH = './data'
count = 50
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
leap = 1

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=False, device = device)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if mtcnn(frame) is not None and leap%2:
        path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
        face_img = mtcnn(frame, save_path = path)
        print('Image captured saved to:', path)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        # print(usr)
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            # print('smt')
            embeds.append(model(trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))
print(names)