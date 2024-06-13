import sys
import cv2
from PyQt6 import uic,QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.model = YOLO('yolov8n.pt')
        self.running_inference  = False

        # Set up the camera capture
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit()

        # Get the video frame size
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video frame size: {self.frame_width} x {self.frame_height}")

        # Resize the QLabel to fit the video frame size
        self.video.setFixedSize(self.frame_width, self.frame_height)

        # Resize the main window to fit the video frame size plus some padding
        window_width = self.frame_width + 20
        window_height = self.frame_height + 170
        self.setFixedSize(window_width, window_height)

        # Set up a timer to update the frame
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Connect buttons to functions
        self.start.clicked.connect(self.start_inference)
        self.stop.clicked.connect(self.stop_inference)
        
    def start_inference(self):
        self.running_inference = True

    def stop_inference(self):
        self.running_inference = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize image
            small_frame = cv2.resize(frame, (640, 480))

            if self.running_inference:
                # Run YOLOv8 inference on the frame
                results = self.model(small_frame,classes=[0])

                # Visualize the results on the frame
                small_frame = results[0].plot()

            # Convert the frame to a format suitable for QLabel
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = small_frame.shape
            step = channel * width
            qimg = QtGui.QImage(small_frame.data, width, height, step, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.video.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
