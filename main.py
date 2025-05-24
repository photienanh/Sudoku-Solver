from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QMessageBox, QHBoxLayout, QFrame
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt
import cv2
import sys
import torch
from detect_board import detect_board, split_cells
from solve_sudoku import get_board, solve_sudoku, draw_solution_on_image, load_model

class SudokuApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sudoku Solver")
        self.setGeometry(100, 100, 700, 800)
        self.setFixedSize(700, 800)

        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f5f6fa"))
        self.setPalette(palette)

        # Title label
        self.title_label = QLabel("Sudoku Solver", self)
        self.title_label.setFont(QFont("Arial", 22, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #273c75; margin-bottom: 10px;")

        # Image display label
        self.label = QLabel(self)
        self.label.setFixedSize(600, 600)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFrameShape(QFrame.Box)
        self.label.setStyleSheet("background-color: white; border: 2px solid #dcdde1; border-radius: 10px;")

        # Buttons
        self.btn_load = QPushButton("Chọn ảnh", self)
        self.btn_load.setFont(QFont("Arial", 12))
        self.btn_load.setFixedWidth(180) 
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #00a8ff;
                color: white;
                border-radius: 8px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #0097e6;
            }
        """)
        self.btn_load.clicked.connect(self.load_image)

        self.btn_solve = QPushButton("Giải Sudoku", self)
        self.btn_solve.setFont(QFont("Arial", 12))
        self.btn_solve.setFixedWidth(180) 
        self.btn_solve.setStyleSheet("""
            QPushButton {
                background-color: #44bd32;
                color: white;
                border-radius: 8px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #4cd137;
            }
        """)
        self.btn_solve.clicked.connect(self.solve_sudoku)

        self.btn_capture = QPushButton("Chụp ảnh", self)
        self.btn_capture.setFont(QFont("Arial", 12))
        self.btn_capture.setFixedWidth(180) 
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: #e17055;
                color: white;
                border-radius: 8px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.btn_capture.clicked.connect(self.capture_image)

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_load)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.btn_solve)
        btn_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.label, alignment=Qt.AlignCenter)
        main_layout.addSpacing(20)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

        self.setLayout(main_layout)

        self.image_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Lỗi", "Không mở được webcam!")
            return
        QMessageBox.information(self, "Hướng dẫn", "Nhấn SPACE để chụp, ESC để hủy.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Chup anh Sudoku", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 32:  # SPACE
                img_path = "captured_sudoku.jpg"
                cv2.imwrite(img_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                self.image_path = img_path
                self.display_image(img_path)
                break
            
    def display_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def solve_sudoku(self):
        if not self.image_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ảnh trước!")
            return
        try:
            img = cv2.imread(self.image_path, 0)
            warp = detect_board(img)
            if warp is None:
                QMessageBox.warning(self, "Lỗi", "Không phát hiện được bảng Sudoku. Vui lòng chụp lại ảnh!")
                return
            cells = split_cells(warp)
            board = get_board(cells, self.model, self.device)
            solved_board = solve_sudoku(board)
            if solved_board is None:
                QMessageBox.warning(self, "Lỗi", "Không tìm được lời giải cho Sudoku này!")
                return
            result_img = draw_solution_on_image(warp.copy(), board, solved_board)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SudokuApp()
    window.show()
    sys.exit(app.exec_())