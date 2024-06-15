import sys
import cv2
from PyQt6 import uic, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow,QLabel
from ultralytics import YOLO
import numpy as np
class ClickableLabel(QLabel):
    clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())

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
        self.video= ClickableLabel(self.video)
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

        # Connect the click signal to a slot
        self.video.clicked.connect(self.on_click)

        # Define multiple rectangle regions of interest (ROI)
        self.rectangles = [
            ((100, 200), (170, 270)), # 가로 : 70, 세로 : 70
            ((70, 230), (140, 300)), # 가로 : 75, 세로 : 70
            ((180, 200), (250, 270)), # 가로 : 75, 세로 : 70
            ((150, 270), (220, 340)), # 가로 : 70, 세로 : 130
            ((90, 300), (160, 370)), # 가로 : 95, 세로 : 140
            # 2조
            ((255, 190), (325, 280)), # 가로 : 70, 세로 : 70
            ((255, 260), (325, 330)), # 가로 : 70, 세로 : 80
            ((325, 190), (395, 280)), # 가로 : 75, 세로 : 70
            ((325, 260), (395, 330)), # 가로 : 75, 세로 : 130
            ((300, 310), (360, 380)), # 가로 : 60, 세로 : 110
            # 3조
            ((400, 200), (470, 270)), # 가로 : 90, 세로 : 70
            ((425, 280), (495, 350)), # 가로 : 75, 세로 : 120
            ((470, 200), (540, 270)), # 가로 : 55, 세로 : 70
            ((520, 230), (590, 300)), # 가로 : 60, 세로 : 70
            ((510, 300), (580, 370)), # 가로 : 60, 세로 : 110
        ]

        self.labels = [
            self.g1_1, self.g1_2, self.g1_3, self.g1_4, self.g1_5,
            self.g2_1, self.g2_2, self.g2_3, self.g2_4, self.g2_5,
            self.g3_1, self.g3_2, self.g3_3, self.g3_4, self.g3_5,
        ]
    def on_click(self, x, y):
        print(f"Mouse clicked at ({x}, {y})")

    def start_inference(self):
        self.running_inference = True

    def stop_inference(self):
        self.running_inference = False

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            large_frame = cv2.resize(frame, (self.output_width, self.output_height))

            if self.running_inference:
                results = self.model(large_frame, classes=[0])
                large_frame = results[0].plot()

                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                # 가상의 바운딩 박스 추가 (여러 명의 사람을 시뮬레이션)
                # additional_boxes = np.array([
                #     [120, 220, 160, 260],  # 사람 1
                #     [275, 210, 315, 270],  # 사람 2
                #     [350, 250, 390, 310],  # 사람 3
                # ])
                # boxes = np.vstack((boxes, additional_boxes))

                # for box in additional_boxes:
                #     cv2.rectangle(large_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                seat_probabilities = []

                for box in boxes:
                    max_iou = 0
                    best_seat_idx = -1
                    for idx, (top_left, bottom_right) in enumerate(self.rectangles):
                        rect_box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                        iou = self.calculate_iou(box, rect_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_seat_idx = idx
                    if max_iou > 0:  # IoU가 0보다 큰 경우에만 추가
                        seat_probabilities.append(best_seat_idx)

                # 중복 제거 및 사람 수만큼의 좌석 선택
                seat_probabilities = list(set(seat_probabilities))

                for idx, label in enumerate(self.labels):
                    if idx in seat_probabilities:
                        label.setStyleSheet("""
                            QLabel {
                                background-color: rgb(255, 0, 0);  /* 배경색: 빨간색 */
                                border: 2px solid black;           /* 테두리: 2px 검정색 실선 */
                                border-radius: 10px;               /* 모서리 둥글게: 반지름 10px */
                            }
                        """)
                    else:
                        label.setStyleSheet("""
                            QLabel {
                                background-color: rgb(0, 255, 0);  /* 배경색: 녹색 */
                                border: 2px solid black;           /* 테두리: 2px 검정색 실선 */
                                border-radius: 10px;               /* 모서리 둥글게: 반지름 10px */
                            }
                        """)

            for top_left, bottom_right in self.rectangles:
                cv2.rectangle(large_frame, top_left, bottom_right, (255, 0, 0), 2)
            

            small_frame = cv2.resize(large_frame, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
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
