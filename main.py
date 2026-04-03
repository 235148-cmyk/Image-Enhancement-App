import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os

# =============================
# GLOBAL VARIABLES
# =============================
img = None
gray = None
output_path = ""

# =============================
# UI STATUS UPDATE
# =============================
def set_status(msg):
    status_label.config(text=msg)

# =============================
# SELECT IMAGE
# =============================
def load_image():
    global img, gray

    path = filedialog.askopenfilename()
    if not path:
        return

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    display_image(img)
    set_status(" Image Loaded Successfully")

# =============================
# SELECT OUTPUT FOLDER
# =============================
def select_output():
    global output_path
    output_path = filedialog.askdirectory()

    if output_path:
        output_path += "/"
        os.makedirs(output_path, exist_ok=True)
        set_status(" Output Folder Selected")

# =============================
# DISPLAY IMAGE
# =============================
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((350, 300))

    imgtk = ImageTk.PhotoImage(image)
    panel.config(image=imgtk)
    panel.image = imgtk

# =============================
# CHECKS
# =============================
def check_ready():
    if img is None:
        set_status(" Load image first")
        return False
    if output_path == "":
        set_status(" Select output folder")
        return False
    return True

# =============================
# SAMPLING
# =============================
def sampling():
    if not check_ready(): return

    scales = [0.5, 0.25, 1, 1.5, 2]
    for s in scales:
        resized = cv2.resize(gray, None, fx=s, fy=s)
        cv2.imwrite(output_path + f"sampling_{s}.jpg", resized)

    set_status(" Sampling done")

# =============================
# QUANTIZATION
# =============================
def quantization():
    if not check_ready(): return

    for bits in [8, 4, 2]:
        levels = 2 ** bits
        quant = np.floor(gray / (256 / levels)) * (256 / levels)
        quant = quant.astype(np.uint8)

        cv2.imwrite(output_path + f"quant_{bits}bit.jpg", quant)

    set_status(" Quantization done")

# =============================
# TRANSFORMATIONS
# =============================
def transformations():
    if not check_ready(): return

    rows, cols = gray.shape

    for angle in [30, 45, 60, 90, 120, 150, 180]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rot = cv2.warpAffine(gray, M, (cols, rows))
        cv2.imwrite(output_path + f"rotation_{angle}.jpg", rot)

    trans = cv2.warpAffine(gray, np.float32([[1,0,50],[0,1,50]]), (cols, rows))
    cv2.imwrite(output_path + "translation.jpg", trans)

    shear = cv2.warpAffine(gray, np.float32([[1,0.5,0],[0,1,0]]), (cols, rows))
    cv2.imwrite(output_path + "shearing.jpg", shear)

    set_status(" Transformations done")

# =============================
# INTENSITY
# =============================
def intensity():
    if not check_ready(): return

    neg = 255 - gray
    cv2.imwrite(output_path + "negative.jpg", neg)

    c = 255 / np.log(1 + np.max(gray))
    log = c * np.log(gray + 1)
    log = np.array(log, dtype=np.uint8)
    cv2.imwrite(output_path + "log.jpg", log)

    gamma1 = np.array(255 * (gray / 255) ** 0.5, dtype='uint8')
    gamma2 = np.array(255 * (gray / 255) ** 1.5, dtype='uint8')

    cv2.imwrite(output_path + "gamma_05.jpg", gamma1)
    cv2.imwrite(output_path + "gamma_15.jpg", gamma2)

    set_status(" Intensity transforms done")

# =============================
# HISTOGRAM
# =============================
def histogram():
    if not check_ready(): return

    plt.hist(gray.ravel(), 256, [0,256])
    plt.savefig(output_path + "hist_original.png")
    plt.close()

    eq = cv2.equalizeHist(gray)

    plt.hist(eq.ravel(), 256, [0,256])
    plt.savefig(output_path + "hist_equalized.png")
    plt.close()

    cv2.imwrite(output_path + "equalized.jpg", eq)

    set_status(" Histogram done")

# =============================
# FINAL PIPELINE
# =============================
def final_pipeline():
    if not check_ready(): return

    gamma = np.array(255 * (gray / 255) ** 0.5, dtype='uint8')
    eq = cv2.equalizeHist(gamma)
    blur = cv2.GaussianBlur(eq, (5,5), 0)

    cv2.imwrite(output_path + "final_enhanced.jpg", blur)
    display_image(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR))

    set_status(" Final Enhancement Done")

# =============================
# RUN ALL
# =============================
def run_all():
    sampling()
    quantization()
    transformations()
    intensity()
    histogram()
    final_pipeline()
    set_status(" All Processing Completed!")

# =============================
# GUI SETUP
# =============================
root = Tk()
root.title("Smart Image Enhancement System")
root.geometry("500x650")

style = ttk.Style()
style.theme_use("clam")

# Title
Label(root, text="Smart Image Enhancement", font=("Arial", 16, "bold")).pack(pady=10)

# Image panel
panel = Label(root)
panel.pack(pady=10)

# Buttons Frame
frame = Frame(root)
frame.pack(pady=10)

ttk.Button(frame, text="Load Image", command=load_image).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(frame, text=" Output Image Path", command=select_output).grid(row=0, column=1, padx=5, pady=5)

ttk.Button(frame, text="Sampling", command=sampling).grid(row=1, column=0, padx=5, pady=5)
ttk.Button(frame, text="Quantization", command=quantization).grid(row=1, column=1, padx=5, pady=5)

ttk.Button(frame, text="Transformations", command=transformations).grid(row=2, column=0, padx=5, pady=5)
ttk.Button(frame, text="Intensity", command=intensity).grid(row=2, column=1, padx=5, pady=5)

ttk.Button(frame, text="Histogram", command=histogram).grid(row=3, column=0, padx=5, pady=5)
ttk.Button(frame, text="Final Enhance", command=final_pipeline).grid(row=3, column=1, padx=5, pady=5)

ttk.Button(root, text=" Run Full Pipeline", command=run_all).pack(pady=15)

# Status bar
status_label = Label(root, text="Ready", bd=1, relief=SUNKEN, anchor=W)
status_label.pack(side=BOTTOM, fill=X)

root.mainloop()