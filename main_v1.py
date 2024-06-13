import sys
import cv2
from PyQt6 import uic, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.model = YOLO('yolov8x.pt')
        self.running_inference = False

        # Set up the camera capture
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit()

        self.max_width = 1920
        self.max_height = 1080

        # Set the video capture resolution to the maximum supported by the camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)

        # Set the output frame size for displaying
        self.output_width = 640
        self.output_height = 480

        # Resize the QLabel to fit the output frame size
        self.video.setFixedSize(self.output_width, self.output_height)

        # Resize the main window to fit the video frame size plus some padding
        window_width = self.output_width + 20
        window_height = self.output_height + 170
        self.setFixedSize(window_width, window_height)

        # Set up a timer to update the frame
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Connect buttons to functions
        self.start.clicked.connect(self.start_inference)
        self.stop.clicked.connect(self.stop_inference)

        # Define multiple rectangle regions of interest (ROI)
        self.rectangles = [
            ((50, 200), (230, 400)),
            ((250, 200), (390, 400)),
            ((400, 200), (570, 400)),
        ]

    def start_inference(self):
        self.running_inference = True

    def stop_inference(self):
        self.running_inference = False

    def is_point_in_rectangles(self, point):
        center_x, center_y = point
        for top_left, bottom_right in self.rectangles:
            if top_left[0] <= center_x <= bottom_right[0] and top_left[1] <= center_y <= bottom_right[1]:
                print(f"Person detected within the rectangle at {top_left} to {bottom_right}!")
                return True
        return False

    def calculate_iou(self, boxA, boxB):
        # Calculate the Intersection over Union (IoU) of two bounding boxes.
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def is_object_in_rectangles_by_iou(self, box):
        for top_left, bottom_right in self.rectangles:
            rect_box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            iou = self.calculate_iou(box, rect_box)
            if iou > 0.5:  # IoU threshold
                print(f"Person detected within the rectangle at {top_left} to {bottom_right} with IoU {iou:.2f}!")
                return True
        return False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize image to the output resolution for processing
            large_frame = cv2.resize(frame, (self.output_width, self.output_height))

            if self.running_inference:
                # Run YOLOv8 inference on the frame
                results = self.model(large_frame, classes=[0])

                # Visualize the results on the frame
                large_frame = results[0].plot()

                # Check for objects within the rectangle ROI
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    self.is_point_in_rectangles((center_x, center_y))
                    self.is_object_in_rectangles_by_iou((x1, y1, x2, y2))

            # Draw the rectangle regions of interest (ROI)
            for top_left, bottom_right in self.rectangles:
                cv2.rectangle(large_frame, top_left, bottom_right, (255, 0, 0), 2)

            # Resize the frame to the output resolution for displaying
            small_frame = cv2.resize(large_frame, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)

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
