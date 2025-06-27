import cv2 
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import os
import torch
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import csv
import datetime
import os.path
import json

root = tk.Tk()
root.geometry("400x300")
root.title('Face Recognition')
proceed_flag = False
total_time = 0
cwd = os.getcwd()
facepath = os.path.join(cwd,"Files")
attendanceFile = 'Attendance.csv'
if not os.path.isdir(facepath):
    os.mkdir(facepath)
def validate(entry, entry2, entry3):
    if entry.get().strip() == "":
        return [0, "Name cannot be empty"]
    elif not entry2.get().strip().isdigit():
        return [0, "Age should be a number"]
    elif entry3.get().strip() == "":
        return [0, "ID cannot be empty"]
    else:
        return [1, "Valid"]
def create_user():
    reset_to_main_menu()
    frame1 = tk.Frame(root)
    frame1.pack(anchor='w', pady=5, padx=5)
    label = tk.Label(frame1, text="Enter full name:")
    label.pack()
    entry = tk.Entry(frame1)
    entry.pack(side=tk.LEFT, anchor='w', padx=5)

    frame2 = tk.Frame(root)
    frame2.pack(anchor='w', pady=5, padx=5)
    label2 = tk.Label(frame2, text="Enter age:")
    label2.pack()
    entry2 = tk.Entry(frame2)
    entry2.pack(side=tk.LEFT, anchor='w', padx=5)

    frame3 = tk.Frame(root)
    frame3.pack(anchor='w', pady=5, padx=5)
    label3 = tk.Label(frame3, text="Enter ID:")
    label3.pack()
    entry3 = tk.Entry(frame3)
    entry3.pack(side=tk.LEFT, anchor='w', padx=5)

    frame4 = tk.Frame(root)
    frame4.pack(anchor='w', padx=5)
    error_label = tk.Label(frame4, text="", fg="red")
    error_label.pack(side=tk.LEFT)
    def validate_and_continue():
        validation = validate(entry, entry2, entry3)
        if validation[0] == 0:
            error_label.config(text=validation[1])
        else:
            error_label.config(text="")
            take_snap(entry, [entry.get(), entry2.get(), entry3.get()])

    frame5 = tk.Frame(root)
    frame5.pack(anchor='w', padx=5)
    btn3 = tk.Button(frame5, text="Enter", command=validate_and_continue)
    btn3.pack(side=tk.LEFT)

def button_click():
    global proceed_flag
    proceed_flag = True

def take_snap(entry, biodata):
    global proceed_flag
    frame6 = tk.Frame(root)
    frame6.pack(anchor='w', padx=5)
    btn4 = tk.Button(frame6, text="Capture", command=button_click)
    btn4.pack(side=tk.LEFT)
    mtcnn, resnet, webcam = initialize_neuralnet()
    take_snap_actual(mtcnn, resnet, webcam, biodata)

def take_snap_actual(mtcnn, resnet, webcam, biodata):
    global proceed_flag, facepath
    check, frame = webcam.read()
    cv2.imshow("Capturing", frame)
    full_name = biodata[0]
    age = biodata[1]
    ID = biodata[2]
    try:
        if not proceed_flag:
            root.after(1000, take_snap_actual, mtcnn, resnet, webcam, biodata)
        else:
            img_embedding = get_embedding(mtcnn, resnet, frame)
    except(Exception):
        messagebox.showinfo("Error", "Error")
        webcam.release()
        cv2.destroyAllWindows()
        reset_to_main_menu()
        return
    if proceed_flag:
        torch.save(img_embedding, os.path.join(facepath, full_name + "_" + age + "_" + ID + '.pt'))
        webcam.release()
        cv2.destroyAllWindows()
        reset_to_main_menu()
        messagebox.showinfo("Title", "Saved successfully")

def recognize_face():
    global total_time
    total_time = 0
    mtcnn, resnet, webcam = initialize_neuralnet()
    recognize_face_actual(mtcnn, resnet, webcam)

def recognize_face_actual(mtcnn, resnet, webcam):
    global total_time, facepath, attendanceFile
    check, frame = webcam.read()
    cv2.imshow("Capturing", frame)
    done = False
    while True:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_cropped = mtcnn(Image.fromarray(rgb))
            if img_cropped is not None:
                img_embedding = resnet(img_cropped.unsqueeze(0))
                files = os.listdir(facepath)
                for f in files:
                    if f.endswith('.pt'):
                        embedding = torch.load(os.path.join(facepath, f))
                        dist = torch.dist(embedding, img_embedding)
                        if dist < 0.85:
                            data = f.split(".")[0].split('_')
                            name = data[0]
                            messagebox.showinfo("Title", "Recognized: " + name)
                            done = True

                            if not os.path.isfile(attendanceFile):
                                writeHeaders()
                            datetime2 = writeAttendanceData(name, data)
                            sendEmail(name, datetime2)
                            break
            if not done:
                if total_time > 5000:
                    sendEmail("abc", datetime.datetime.now())
                    messagebox.showinfo("Error", "Terminating as no face recognized")
                    cv2.destroyAllWindows()
                    reset_to_main_menu()
                    webcam.release()
                    break

                total_time += 1000
                root.after(1000, recognize_face_actual, mtcnn, resnet, webcam)
            else:
                cv2.destroyAllWindows()
                reset_to_main_menu()
                webcam.release()
            break
        except Exception as e:
            messagebox.showinfo("Error", e)
            cv2.destroyAllWindows()
            reset_to_main_menu()
            webcam.release()
            break
def writeHeaders():
    global attendanceFile
    newRows = [["Name", "Age", "ID", "Date & Time"]]
    file2 = open(attendanceFile, 'w', newline='')
    writer = csv.writer(file2)
    writer.writerows(newRows)
    file2.close()
def writeAttendanceData(name, data):
    time = datetime.datetime.now()
    newRows = [[name, data[1], data[2], time]]
    file2 = open(attendanceFile, 'a', newline='')
    writer = csv.writer(file2)
    writer.writerows(newRows)
    file2.close()
    return time
def sendEmail(name, datetime):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText
    from email.header import Header
    msg = MIMEMultipart()
    msg.set_charset('utf8')
    msg['Subject'] = 'Attendance of ' + name
    with open('config.json', 'r') as fp:
        config = json.load(fp)

    msg['From'] = config['From']
    msg['To'] = config['To']
    body_text = f'User {name} has clocked in at {str(datetime)}\n Regards,\n MindsEye © Reality Krafters'

    body = MIMEText(body_text, 'plain', 'utf-8')
    msg.attach(body)

    file = 'MindsEye.png'
    with open(file, 'rb') as fp:
        img = MIMEImage(fp.read())
        img.add_header('Content-Disposition', "attachment; filename= MindsEye")
    msg.attach(img)

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(config['From'],config['SMTPPassword'])
    output = server.send_message(msg, msg['From'], msg['To'])
    server.quit()

def initialize_neuralnet():
    mtcnn = MTCNN(image_size=200)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    webcam = cv2.VideoCapture(0)
    return mtcnn, resnet, webcam
def get_embedding(mtcnn, resnet, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_cropped = mtcnn(Image.fromarray(rgb))
    img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding

def reset_to_main_menu():
    for widget in root.winfo_children():
        if widget not in [top_frame]:
            widget.destroy()

tree = None
def show_history():
    global tree
    if tree and tree.winfo_exists():
        tree.destroy()
    rows = []
    try:
        with open('Attendance.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            rows.pop(0)
            rows = [row for row in rows if len(row) == 4]
    except(FileNotFoundError):
        messagebox.showinfo("Error", "Attendance file not found")
        return

    tree = ttk.Treeview(root, columns=('Name','Age','ID','Date & Time'), show='headings')
    tree.heading('Name', text='Name', command=lambda: sort_by('Name',False))
    tree.heading('Age', text='Age', command=lambda: sort_by('Age',False))
    tree.heading('ID', text='ID', command=lambda: sort_by('ID',False))
    tree.heading('Date & Time', text='Date & Time', command=lambda: sort_by('Date & Time',False))
    tree.pack()
    for row in rows:
        tree.insert('', tk.END, values=row)

    def sort_by(col, descending):
        idx = {'Name': 0, 'Age': 1, 'ID': 2, 'Date & Time': 3}[col]
        sorted_data = sorted(rows, key=lambda x: x[idx], reverse=descending)
        tree.delete(*tree.get_children())
        for row in sorted_data:
            tree.insert('', tk.END, values=row)
        tree.heading(col, text=f"{col} {'▼' if descending else '▲'}",
                     command=lambda: sort_by(col, not descending))

top_frame = tk.Frame(root)
top_frame.pack(anchor='w', pady=5, padx=5)
btn1 = tk.Button(top_frame, text="Create new user", command=create_user)
btn1.pack(side=tk.LEFT, padx=5)
btn2 = tk.Button(top_frame, text="Recognize face", command=recognize_face)
btn2.pack(side=tk.LEFT, padx=5)
btn3 = tk.Button(top_frame, text="Show history", command=show_history)
btn3.pack(side=tk.LEFT, padx=5)
root.mainloop()