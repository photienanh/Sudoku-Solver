import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
import os
from detect_board import detect_board, split_cells
from solve_sudoku import get_board, solve_sudoku, draw_solution_on_image, load_model

def process_image(image):
    """
    Xử lý ảnh Sudoku: phát hiện bảng, nhận dạng số và giải
    
    Args:
        image: PIL Image - ảnh đầu vào
        
    Returns:
        tuple: (result_image, status_message)
    """
    try:
        if image is None:
            return None, "⚠️ Vui lòng chọn hoặc chụp ảnh trước!"
        
        # Chuyển đổi PIL image sang định dạng OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Chuyển sang ảnh xám
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện và trích xuất bảng Sudoku
        warp = detect_board(gray_image)
        if warp is None:
            return image, "❌ Không phát hiện được bảng Sudoku. Vui lòng chụp lại ảnh rõ nét hơn!"
        
        # Tách bảng thành 81 ô nhỏ
        cells = split_cells(warp)
        if cells is None:
            return image, "❌ Không thể tách các ô trong bảng Sudoku"

        # Khởi tạo device cho PyTorch (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tải model nhận dạng chữ số
        model = load_model(device)
        if model is None:
            return image, "❌ Không thể tải model nhận dạng số"
        
        # Nhận dạng các số trong bảng Sudoku
        board = get_board(cells, model, device)
        
        # Giải bảng Sudoku
        solved_board = solve_sudoku(board)
        if solved_board is None:
            return image, "❌ Ảnh chụp bị mờ hoặc không thể giải được bảng Sudoku này. Vui lòng chụp lại!"
        
        # Vẽ lời giải lên ảnh gốc
        result_image = draw_solution_on_image(warp.copy(), board, solved_board)
        
        # Chuyển đổi từ BGR sang RGB để hiển thị
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        return result_pil, "✅ Đã giải thành công!"
        
    except Exception as e:
        error_msg = f"❌ Đã xảy ra lỗi: {str(e)}"
        import traceback
        traceback.print_exc()
        return image, error_msg

def main():
    """Hàm chính khởi tạo giao diện Gradio"""
    
    # CSS tùy chỉnh cho giao diện
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif !important;
        background-color: #f5f6fa !important;
        max-width: 700px !important;
        margin: 0 auto !important;
    }
    
    .title-header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #273c75 !important;
        text-align: center !important;
        margin-bottom: 10px !important;
        padding: 20px !important;
    }
    
    .main-image {
        width: 600px !important;
        height: 600px !important;
        background-color: white !important;
        border: 2px solid #dcdde1 !important;
        border-radius: 10px !important;
        margin: 0 auto !important;
    }
    
    .btn-load {
        background-color: #00a8ff !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 16px 32px !important;
        width: 220px !important;
        font-family: 'Arial' !important;
        font-size: 15px !important;
        border: none !important;
        margin: 0 10px !important;
        font-weight: bold !important;
        cursor: pointer;
    }

    .btn-load:hover {
        background-color: #0097e6 !important;
    }

    .btn-capture {
        background-color: #e17055 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 16px 32px !important;
        width: 220px !important;
        font-family: 'Arial' !important;
        font-size: 15px !important;
        border: none !important;
        margin: 0 10px !important;
        font-weight: bold !important;
        cursor: pointer;
    }

    .btn-capture:hover {
        background-color: #d35400 !important;
    }

    .btn-solve {
        background-color: #44bd32 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 16px 32px !important;
        width: 220px !important;
        font-family: 'Arial' !important;
        font-size: 15px !important;
        border: none !important;
        margin: 0 10px !important;
        font-weight: bold !important;
        cursor: pointer;
    }

    .btn-solve:hover {
        background-color: #4cd137 !important;
    }

    .button-row {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 20px !important;
        margin: 20px 0 !important;
    }
    
    .status-text {
        text-align: center !important;
        padding: 10px !important;
        margin: 10px 0 !important;
        background-color: white !important;
        border-radius: 5px !important;
        border: 1px solid #dcdde1 !important;
    }
    
    .camera-container {
        width: 600px !important;
        height: 600px !important;
        background-color: white !important;
        border: 2px solid #dcdde1 !important;
        border-radius: 10px !important;
        margin: 0 auto !important;
    }
    
    /* Tắt tất cả transform để không lật ảnh camera */
    .camera-container .wrap {
        transform: none !important;
    }
    
    .camera-container video {
        transform: none !important;
    }
    
    .camera-container canvas {
        transform: none !important;
    }
    """
    
    # Tạo giao diện Gradio
    with gr.Blocks(
        title="Sudoku Solver",
        css=custom_css,
        theme=gr.themes.Default()
    ) as demo:
        
        # Tiêu đề ứng dụng
        gr.HTML("""
            <div class="title-header">
                <h1 style="font-size: 36px; font-weight: bold; color: #273c75; margin: 0; padding: 0;">Sudoku Solver</h1>
            </div>
        """)
        
        # Khu vực hiển thị chính
        with gr.Column():
            # Hiển thị ảnh
            image_display = gr.Image(
                type="pil",
                label="",
                height=600,
                width=600,
                elem_classes="main-image",
                show_label=False,
                interactive=False,
                visible=True
            )
            
            # Hiển thị camera (ẩn mặc định)
            camera_display = gr.Image(
                type="pil",
                label="📸 Camera",
                height=600,
                width=600,
                elem_classes="camera-container",
                show_label=False,
                sources=["webcam"],
                interactive=True,
                visible=False,
                mirror_webcam=False  # Tắt mirror để ảnh không bị lật
            )
        
        # Hàng 3 nút chức năng
        with gr.Row(elem_classes="button-row"):
            btn_load = gr.UploadButton(
                "📁 Chọn ảnh",
                file_types=["image"],
                elem_classes="btn-load",
                size="sm"
            )
            
            btn_capture = gr.Button(
                "📸 Chụp ảnh",
                elem_classes="btn-capture",
                size="sm"
            )
            
            btn_solve = gr.Button(
                "🚀 Giải Sudoku",
                elem_classes="btn-solve",
                size="sm"
            )
        
        # Thông báo trạng thái (ẩn mặc định)
        status_output = gr.Textbox(
            label="",
            visible=False,
            elem_classes="status-text"
        )
        
        # Biến state để lưu trạng thái
        current_image = gr.State(None)
        camera_mode = gr.State(False)
        
        # Xử lý sự kiện tải ảnh từ file
        def load_image_handler(file):
            """Xử lý khi người dùng tải ảnh từ file"""
            if file is not None:
                image = Image.open(file.name)
                return (
                    image,  # Hiển thị ảnh
                    gr.update(visible=False),  # Ẩn camera
                    image,  # Lưu ảnh vào state
                    False,  # Tắt camera mode
                    "",  # Xóa status
                    gr.update(visible=False),  # Ẩn status
                    "Chụp ảnh"  # Reset text nút
                )
            return None, gr.update(), None, False, "", gr.update(visible=False), "Chụp ảnh"
        
        # Xử lý bật/tắt camera
        def toggle_camera(is_camera_mode):
            """Chuyển đổi giữa chế độ camera và hiển thị ảnh"""
            if is_camera_mode:
                # Tắt camera, chuyển về hiển thị ảnh
                return (
                    gr.update(visible=True),   # Hiện ảnh
                    gr.update(visible=False),  # Ẩn camera
                    False,  # Tắt camera mode
                    "Chụp ảnh"  # Đổi text nút
                )
            else:
                # Bật camera, ẩn hiển thị ảnh
                return (
                    gr.update(visible=False),  # Ẩn ảnh
                    gr.update(visible=True),   # Hiện camera
                    True,   # Bật camera mode
                    "Đóng camera"  # Đổi text nút
                )
        
        # Xử lý chụp ảnh từ camera
        def capture_handler(webcam_img, is_camera_mode):
            """Xử lý khi người dùng chụp ảnh từ camera"""
            if webcam_img is not None and is_camera_mode:
                return (
                    gr.update(value=webcam_img, visible=True),  # Hiển thị ảnh vừa chụp
                    gr.update(visible=False),  # Ẩn camera
                    webcam_img,  # Lưu ảnh vào state
                    False,  # Tắt camera mode
                    "📸 Đã chụp ảnh từ camera!",  # Thông báo
                    gr.update(visible=True),  # Hiện thông báo
                    "Chụp ảnh"  # Reset text nút
                )
            return None, gr.update(), None, is_camera_mode, "", gr.update(visible=False), "Chụp ảnh"
        
        # Xử lý giải Sudoku
        def solve_handler(current_img):
            """Xử lý khi người dùng nhấn nút giải Sudoku"""
            if current_img is None:
                return None, "⚠️ Vui lòng chọn hoặc chụp ảnh trước!", gr.update(visible=True)
            
            # Gọi hàm xử lý ảnh
            result_img, status = process_image(current_img)
            
            # Hiển thị kết quả và thông báo
            if "❌" in status:
                return result_img, status, gr.update(visible=True)
            
            return result_img, "", gr.update(visible=False)
        
        # Kết nối các sự kiện với hàm xử lý
        btn_load.upload(
            fn=load_image_handler,
            inputs=btn_load,
            outputs=[image_display, camera_display, current_image, camera_mode, status_output, status_output, btn_capture]
        )
        
        btn_capture.click(
            fn=toggle_camera,
            inputs=camera_mode,
            outputs=[image_display, camera_display, camera_mode, btn_capture]
        )
        
        camera_display.change(
            fn=capture_handler,
            inputs=[camera_display, camera_mode],
            outputs=[image_display, camera_display, current_image, camera_mode, status_output, status_output, btn_capture]
        )
        
        btn_solve.click(
            fn=solve_handler,
            inputs=current_image,
            outputs=[image_display, status_output, status_output]
        )
    
    # Khởi chạy ứng dụng
    demo.launch()

if __name__ == "__main__":
    main()