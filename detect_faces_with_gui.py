import os
import cv2
import shutil
import tkinter as tk
from tkinter import filedialog
import sys

class TextRedirector(object):
    def __init__(self, widget, terminal):
        self.widget = widget
        self.terminal = terminal

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')
        self.terminal.write(str)

    def flush(self):
        pass

def load_mobilenet_model():
    model_path = 'mobilenet_iter_73000.caffemodel'
    config_path = 'deploy.prototxt'
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net

def count_people_in_image(image_path, net, class_names):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx in class_names and class_names[idx] == 'person':
                person_count += 1

    return person_count

def rename_and_copy_images_in_folder(input_folder, output_folder, backup_folder, net, class_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    image_found = False
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_found = True
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            shutil.copy(image_path, os.path.join(backup_folder, filename))
            print(f"Backup created on backup folder for {filename}")

            person_count = count_people_in_image(image_path, net, class_names)
            print(f"Found {person_count} person(s) in {filename}")
            file_base, file_extension = os.path.splitext(filename)
            new_filename = f"{file_base}_({person_count} personnes){file_extension}"
            new_image_path = os.path.join(output_folder, new_filename)

            shutil.copy(image_path, new_image_path)
            for i in range(1, person_count):
                copy_filename = f"{file_base}_({person_count} personnes)_copy{i}{file_extension}"
                copy_image_path = os.path.join(output_folder, copy_filename)
                shutil.copy(new_image_path, copy_image_path)
            print(f"Created {person_count} copies of {filename} in output folder")

            os.remove(image_path)

    if not image_found:
        print("No pictures found")

def clear_output():
    output_text.configure(state='normal')
    output_text.delete(1.0, tk.END)
    output_text.configure(state='disabled')

def select_folder(label, title):
    initialdir = os.path.dirname(os.path.abspath(__file__))
    folder = filedialog.askdirectory(message=title, initialdir=initialdir)
    label.config(text=folder)
    return folder

def main():
    net = load_mobilenet_model()
    class_names = {15: 'person'}

    root = tk.Tk()
    root.geometry("800x400")
    root.columnconfigure(0, weight=1)
    root.resizable(False, False)
    root.title("Humans detector")

    input_title = "Select Input Folder"
    input_label = tk.Label(root, text="")
    input_label.grid(row=0, column=0, sticky='s')  # 's' means south or bottom
    input_button = tk.Button(root, text=input_title, command=lambda: select_folder(input_label, input_title))
    input_button.grid(row=1, column=0, sticky='ew')  # 'ew' means east and west, or left and right

    output_title = "Select Output Folder"
    output_label = tk.Label(root, text="")
    output_label.grid(row=2, column=0, sticky='s')  # 's' means south or bottom
    output_button = tk.Button(root, text=output_title, command=lambda: select_folder(output_label, output_title))
    output_button.grid(row=3, column=0, sticky='ew')  # 'ew' means east and west, or left and right

    backup_title = "Select Backup Folder"
    backup_label = tk.Label(root, text="")
    backup_label.grid(row=4, column=0, sticky='s')  # 's' means south or bottom
    backup_button = tk.Button(root, text=backup_title, command=lambda: select_folder(backup_label, backup_title))
    backup_button.grid(row=5, column=0, sticky='ew')  # 'ew' means east and west, or left and right

    process_button = tk.Button(root, text="Process", command=lambda: rename_and_copy_images_in_folder(input_label.cget("text"), output_label.cget("text"), backup_label.cget("text"), net, class_names))
    process_button.grid(row=6, column=0, columnspan=2, sticky='s', pady=20)

    global output_text
    output_text = tk.Text(root, state='disabled', height=10)
    output_text.grid(row=7, column=0, sticky='ew')

    sys.stdout = TextRedirector(output_text, sys.stdout)

    clear_button = tk.Button(root, text="Clear Output", command=clear_output)
    clear_button.grid(row=8, column=0, sticky='ew')

    root.mainloop()

if __name__ == "__main__":
    main()
