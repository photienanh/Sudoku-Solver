# SUDOKU SOLVER
A Python project for solving Sudoku puzzles automatically. It uses computer vision and a deep learning model (PyTorch) to detect and solve Sudoku puzzles from images or webcam input.
## Features
- Load Sudoku puzzle from an image file or Capture Sudoku puzzle from webcam.
- Detect and extract Sudoku grid and digits using image processing techniques.
- Use a trained deep learning model to recognize digits.
- Solve the puzzle and display the solution.
- Simple and user-friendly GUI.
## Preview
Below are some example screenshots of the application in use:

| Upload Image | Solved Result | Capture from Webcam | Solved Webcam Result |
|:------------:|:-------------:|:-------------------:|:--------------------:|
| ![Image](https://cdn.anh.moe/f/G6IXpYWn.png-webp) | ![Image](https://cdn.anh.moe/f/Lr5yjM.png-webp) | ![Image](https://cdn.anh.moe/f/xsbnJfZH.png-webp) | ![Image](https://cdn.anh.moe/f/ad1XRo2q.png-webp) |
## How to Run

### Option 1: Online Version (Recommended)
🚀 **Try it instantly without any installation!**

Visit: [https://huggingface.co/spaces/photienanh/Sudoku-Solver](https://huggingface.co/spaces/photienanh/Sudoku-Solver)

### Option 2: Local Installation
## Directory Structure
```
SudokuSolver/
│
├── detect_board.py              # Detects and extracts Sudoku grid and cells from the image
├── digit_dataset_generation.py  # Script to generate digit dataset
├── digit_recognition_train.py   # Trains a CNN model (PyTorch) for digit recognition
├── solve_sudoku.py              # Recognizes digits, solves Sudoku, and draws solution
├── main.py                      # GUI application (PyQt5)
├── requirements.txt             # Required libraries
└── Model/
    └── digit_cnn.pth            # Trained digit recognition model
```
## Requirements
- Python 3.6 or higher.
- Libraries in requirements.txt.
## Installation (For Local Setup)
### 1. Clone the repository
```bash
git clone https://github.com/photienanh/Sudoku-Solver
cd Sudoku-Solver
```
Alternatively, download the ZIP file from GitHub and extract it.
### 2. Install Dependencies
Ensure Python is installed. If not, you can download and install it from the official [Python website](https://www.python.org/downloads/). Then, install the required libraries:
```bash
pip install -r requirements.txt
```
## Usage
1. (Optional) If you want to train your own digit recognition model:
    - Run ```python digit_dataset_generation.py``` to generate the digit dataset for training.
    - Run ```python digit_recognition_train.py``` to train the digit recognition model.
    > **Note:** These steps can be skipped because a pre-trained model (`Model/digit_cnn.pth`) is already included.

2. To start the application:
    - Run `main.py` to open the graphical user interface and begin using the Sudoku Solver.
    ```bash
    python main.py
    ```
### Webcam Instructions
- Press **SPACE** to capture an image of the Sudoku puzzle from your webcam.
- Press **ESC** to cancel.
- Make sure the Sudoku grid is clearly visible for accurate detection.

### Image Upload Instructions
- Select the option to upload an image file containing the Sudoku puzzle.

After selecting the Sudoku image using either method above, click **"Giải Sudoku"** to process and solve the puzzle. The solved Sudoku board will then be displayed clearly in the application.

⚠️ **Notes**
- Ensure the Sudoku grid is well-lit and clear for best results.
- Blurry or handwritten digits may affect digit recognition accuracy.
## Contribution

Feel free to open issues or submit pull requests for improvements or bug fixes!