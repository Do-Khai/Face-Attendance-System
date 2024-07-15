import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from datetime import datetime
import csv
import os

frame_size = (1640,1480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def inference(model, face, local_embeds, threshold = 3):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
    
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face



def save_attendance_record(name, status):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    late_time = now.replace(hour=8, minute=0, second=0, microsecond=0)

    if now > late_time:
        status = "Late"

    if not os.path.exists("attendance.csv") or os.stat("attendance.csv").st_size == 0:
        with open("attendance.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Timestamp", "Status"])

    with open("attendance.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the attendance record to the CSV file
        writer.writerow([name, timestamp, status])


def load_attendance_data():
    if not os.path.exists("attendance.csv"):
        return {}  

    attendance = {}
    with open("attendance.csv", mode="r", newline="") as file:
        reader = csv.reader(file)

        # Bo qua hang dau 
        next(reader)

        for row in reader:
            name, timestamp, status = row
            attendance[name] = float(time.mktime(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple()))

    return attendance


def save_check_in_time(name, status):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    late_time = now.replace(hour=8, minute=0, second=0, microsecond=0)

    if now > late_time:
        status = "Late"
    else:
        status = "On time"

    print(f"{name} checked in at {timestamp} with status: {status}")

    # Them vao file csv neu no trong 
    if not os.path.exists("attendance.csv") or os.stat("attendance.csv").st_size == 0:
        with open("attendance.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Timestamp", "Status"])

    with open("attendance.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        # Viet vao file csv
        writer.writerow([name, timestamp, status])



if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1480)
    embeddings, names = load_faceslist()

    attendance = load_attendance_data()

    attendance_taken = False

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                        x = names[idx]
                        if x not in attendance:
                            attendance[x] = time.time()  # Thoi gian check-in
                        else:
                        # Sau 5 giay se tinh attend 
                            if time.time() - attendance[x] >= 5 and not attendance_taken:
                            # Luu vao file csv
                                save_check_in_time(x, "")
                                attendance[x] = time.time()  # Thoi gian check-in
                                # Them 'Attended' duoi bounding box
                                attendance_taken = True
                                text = "Attended"
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                                frame = cv2.putText(frame, text,
                                                    (bbox[0] + (bbox[2] - bbox[0]) // 2 - text_size[0] // 2,
                                                     bbox[3] + text_size[1] + 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Face Recognition', frame)

            if attendance_taken:
                break
        
        if cv2.waitKey(1)&0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    #+": {:.2f}".format(score)