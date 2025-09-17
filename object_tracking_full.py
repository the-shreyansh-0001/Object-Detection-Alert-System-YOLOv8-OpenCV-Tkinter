import cv2
from ultralytics import YOLO
import winsound
import tkinter as tk
from threading import Thread


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)
running = False


def beep_alert():
    winsound.Beep(1000, 200)  # frequency=1000Hz, duration=200ms


def run_detection():
    global running, cap
    while running:
        ret, frame = cap.read()
        if not ret:
            break


        results = model.track(frame, persist=True)


        if len(results[0].boxes) > 0:
            beep_alert()


        annotated_frame = results[0].plot()


        cv2.imshow("YOLOv8 Object Tracking with Sound", annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


def start_detection():
    global running
    if not running:
        running = True
        Thread(target=run_detection, daemon=True).start()

def stop_detection():
    global running
    running = False



def quit_app():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


root = tk.Tk()
root.title("Object Tracking Control Panel")

start_btn = tk.Button(root, text="▶ Start Detection", command=start_detection, width=20, bg="green", fg="white")
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="⏸ Stop Detection", command=stop_detection, width=20, bg="orange", fg="white")
stop_btn.pack(pady=5)

quit_btn = tk.Button(root, text="❌ Quit", command=quit_app, width=20, bg="red", fg="white")
quit_btn.pack(pady=5)

root.mainloop()
