import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    
    # CSS responsive cho cả PC và Mobile
    custom_css = """
    /* Container chính */
    .gradio-container {
        font-family: 'Arial', sans-serif !important;
        background-color: #f5f6fa !important;
        max-width: 800px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    
    /* Tiêu đề */
    .title-header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #273c75 !important;
        text-align: center !important;
        margin-bottom: 15px !important;
        padding: 20px !important;
    }
    
    /* Hiển thị ảnh */
    .main-image, .camera-container {
        width: 100% !important;
        max-width: 600px !important;
        height: 600px !important;
        background-color: white !important;
        border: 2px solid #dcdde1 !important;
        border-radius: 12px !important;
        margin: 0 auto !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Buttons styling */
    .btn-load, .btn-capture, .btn-solve {
        color: white !important;
        border-radius: 10px !important;
        padding: 14px 20px !important;
        width: 200px !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border: none !important;
        margin: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.16) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .btn-load {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .btn-load:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
    }

    .btn-capture {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }

    .btn-capture:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4) !important;
    }

    .btn-solve {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    }

    .btn-solve:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4) !important;
    }

    /* Button row */
    .button-row {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 15px !important;
        margin: 25px 0 !important;
        flex-wrap: wrap !important;
    }
    
    /* Status text */
    .status-text {
        text-align: center !important;
        padding: 12px !important;
        margin: 15px 0 !important;
        background-color: white !important;
        border-radius: 8px !important;
        border: 1px solid #dcdde1 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Camera settings */
    .camera-container .wrap {
        transform: none !important;
    }
    
    .camera-container video {
        transform: none !important;
    }
    
    .camera-container canvas {
        transform: none !important;
    }
    
    /* ===== RESPONSIVE MOBILE ===== */
    @media (max-width: 768px) {
        /* Container */
        .gradio-container {
            max-width: 100% !important;
            padding: 10px !important;
            margin: 0 !important;
        }
        
        /* Tiêu đề mobile */
        .title-header {
            font-size: 24px !important;
            padding: 15px 10px !important;
            margin-bottom: 10px !important;
        }
        
        /* Ảnh responsive */
        .main-image, .camera-container {
            width: 95% !important;
            height: 350px !important;
            margin: 10px auto !important;
        }
        
        /* Buttons mobile */
        .btn-load, .btn-capture, .btn-solve {
            width: 140px !important;
            padding: 12px 16px !important;
            font-size: 13px !important;
            margin: 5px !important;
        }
        
        /* Button row mobile */
        .button-row {
            gap: 8px !important;
            margin: 15px 0 !important;
            justify-content: space-around !important;
        }
        
        /* Status mobile */
        .status-text {
            font-size: 12px !important;
            padding: 10px !important;
            margin: 10px 5px !important;
        }
    }
    
    /* ===== RESPONSIVE TABLET ===== */
    @media (min-width: 769px) and (max-width: 1024px) {
        .title-header {
            font-size: 30px !important;
        }
        
        .main-image, .camera-container {
            height: 500px !important;
        }
        
        .btn-load, .btn-capture, .btn-solve {
            width: 180px !important;
            font-size: 14px !important;
        }
    }
    
    /* ===== RESPONSIVE LARGE SCREEN ===== */
    @media (min-width: 1200px) {
        .gradio-container {
            max-width: 900px !important;
        }
        
        .main-image, .camera-container {
            height: 650px !important;
        }
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
                mirror_webcam=False
            )
        
        # Hàng 3 nút chức năng
        with gr.Row(elem_classes="button-row"):
            btn_load = gr.UploadButton(
                "📁 Chọn ảnh",
                file_types=["image"],
                elem_classes="btn-load",
                size="lg"
            )
            
            btn_capture = gr.Button(
                "📸 Chụp ảnh",
                elem_classes="btn-capture",
                size="lg"
            )
            
            btn_solve = gr.Button(
                "🚀 Giải Sudoku",
                elem_classes="btn-solve",
                size="lg"
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
                    "📸 Chụp ảnh"  # Reset text nút
                )
            return None, gr.update(), None, False, "", gr.update(visible=False), "📸 Chụp ảnh"
        
        # Xử lý bật/tắt camera
        def toggle_camera(is_camera_mode):
            """Chuyển đổi giữa chế độ camera và hiển thị ảnh"""
            if is_camera_mode:
                # Tắt camera, chuyển về hiển thị ảnh
                return (
                    gr.update(visible=True),   # Hiện ảnh
                    gr.update(visible=False),  # Ẩn camera
                    False,  # Tắt camera mode
                    "📸 Chụp ảnh"  # Đổi text nút
                )
            else:
                # Bật camera, ẩn hiển thị ảnh
                return (
                    gr.update(visible=False),  # Ẩn ảnh
                    gr.update(visible=True),   # Hiện camera
                    True,   # Bật camera mode
                    "❌ Đóng camera"  # Đổi text nút
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
                    "📸 Đã chụp ảnh từ camera thành công!",  # Thông báo
                    gr.update(visible=True),  # Hiện thông báo
                    "📸 Chụp ảnh"  # Reset text nút
                )
            return None, gr.update(), None, is_camera_mode, "", gr.update(visible=False), "📸 Chụp ảnh"
        
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
            
            return result_img, status, gr.update(visible=True)
        
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