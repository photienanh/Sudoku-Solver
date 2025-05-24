import os
import sys
import cv2
import numpy as np
import torch
import copy
from digit_recognition_train import DigitCNN

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def is_empty_cell(cell_img, threshold=0.05, contrast_thresh=15):
    blur = cv2.GaussianBlur(cell_img, (17, 17), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    center = thresh[10:40, 10:40]
    white_ratio = np.sum(center > 0) / center.size

    if white_ratio < threshold:
        std_dev = np.std(center)
        return std_dev < contrast_thresh
    return False


def predict_digit(cell_img, model, device):
    img = cv2.resize(cell_img, (28, 28))

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item() + 1

    return pred

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3*(row//3) + i//3][3*(col//3) + i%3] == num:
            return False
    return True

def solve_sudoku(board):
    def backtrack(b):
        for row in range(9):
            for col in range(9):
                if b[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(b, row, col, num):
                            b[row][col] = num
                            if backtrack(b):
                                return True
                            b[row][col] = 0
                    return False
        return True

    board_copy = copy.deepcopy(board)
    if backtrack(board_copy):
        return board_copy
    else:
        return None

def get_board(cells, model, device):
    board = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            if is_empty_cell(cells[i * 9 + j]):
                board[i, j] = 0
            else:
                board[i, j] = predict_digit(cells[i * 9 + j], model, device)
    return board

def load_model(device):
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(resource_path("Model/digit_cnn.pth")))
    return model

def draw_solution_on_image(warp_img, original_board, solved_board):
    h, w = warp_img.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    if len(warp_img.shape) == 2 or warp_img.shape[2] == 1:
        warp_img = cv2.cvtColor(warp_img, cv2.COLOR_GRAY2BGR)
    for row in range(9):
        for col in range(9):
            if original_board[row][col] == 0:
                num = solved_board[row][col]
                text = str(num)
                x = col * cell_w
                y = row * cell_h
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_x = x + (cell_w - text_size[0]) // 2
                text_y = y + (cell_h + text_size[1]) // 2
                cv2.putText(warp_img, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return warp_img