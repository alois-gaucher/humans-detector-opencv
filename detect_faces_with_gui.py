import cv2
import os
import shutil
import numpy as np

def is_tkinter_available():
    try:
        from tkinter import Tk, filedialog
        return True, Tk, filedialog
    except ImportError:
        return False, None, None

def load_mobilenet_model():
    model_path = 'mobilenet_iter_73000.caffemodel'
    config_path = 'deploy.prototxt'
    return cv2.dnn.readNetFromCaffe(config_path, model_path)

def count_people_in_image(image_path, net, class_names):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # minimum confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx in class_names and class_names[idx] == 'person':
                person_count += 1

    return person_count

def process_image(filename, input_folder, output_folder, backup_folder, net, class_names):
    image_path = os.path.join(input_folder, filename)
    backup_image_path = os.path.join(backup_folder, filename)
    shutil.copy(image_path, backup_image_path)
    print(f"Copied '{filename}' to backup folder")

    person_count = count_people_in_image(image_path, net, class_names)
    file_base, file_extension = os.path.splitext(filename)
    new_filename = f"{file_base}_({person_count} personnes){file_extension}"
    new_image_path = os.path.join(output_folder, new_filename)
    shutil.copy(image_path, new_image_path)
    print(f"Copied and renamed '{filename}' to '{new_filename}'")

    for i in range(1, person_count):
        copy_filename = f"{file_base}_({person_count} personnes)_copy{i}{file_extension}"
        copy_image_path = os.path.join(output_folder, copy_filename)
        shutil.copy(new_image_path, copy_image_path)
        print(f"Created copy '{copy_filename}'")

    os.remove(image_path)
    print(f"Removed '{filename}' from original folder")

def rename_and_copy_images_in_folder(input_folder, output_folder, backup_folder, net, class_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    image_found = False
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            image_found = True
            process_image(filename, input_folder, output_folder, backup_folder, net, class_names)

    if not image_found:
        print("No pictures found")

def main():
    tkinter_available, Tk, filedialog = is_tkinter_available()
    net = load_mobilenet_model()
    class_names = {15: 'person'}

    if tkinter_available:
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        initialdir = os.path.dirname(os.path.abspath(__file__))
        input_folder = filedialog.askdirectory(message="Select Input Folder", initialdir=initialdir)
        output_folder = filedialog.askdirectory(message="Select Output Folder", initialdir=initialdir)
        backup_folder = filedialog.askdirectory(message="Select Backup Folder", initialdir=initialdir)

        if input_folder and output_folder and backup_folder:
            rename_and_copy_images_in_folder(input_folder, output_folder, backup_folder, net, class_names)
        else:
            print("Folder selection cancelled.")
    else:
        print("tkinter is not available. Please install tkinter to use the file dialog feature.")
        # Fallback to hardcoded paths or any other method to provide the paths
        input_folder = './images'
        output_folder = './output'
        backup_folder = './images_backup'
        rename_and_copy_images_in_folder(input_folder, output_folder, backup_folder, net, class_names)

if __name__ == "__main__":
    main()
