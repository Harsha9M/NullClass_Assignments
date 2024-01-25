import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

class EyeDetectionApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Eye Detection App")

        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(pady=10)
        
        self.btn_stop = tk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(pady=5)

        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.is_running = False
        self.update_thread = None

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start(self):
        self.is_running = True
        self.vid = cv2.VideoCapture(self.video_source)
        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()

    def stop(self):
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()

    def on_close(self):
        self.is_running = False
        self.root.destroy()

    def update(self):
        while self.is_running:
            ret, frame = self.vid.read()

            if ret:
                eyes = self.detect_eyes(frame)
                text = "Eyes Open" if eyes else "Eyes Closed"
                self.draw_text(frame, text)

                photo = self.convert_to_photo_image(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            self.root.after(10, lambda: None)  # This is needed to update the GUI
            self.root.update_idletasks()  # Handle events

    def detect_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return len(eyes) > 0

    def draw_text(self, frame, text):
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def convert_to_photo_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=img)
        return photo

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeDetectionApp(root)
    root.mainloop()
