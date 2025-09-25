* Hướng dẫn sử dụng phần mềm python nhận diện cảm xúc trên khuôn mặt:
1.Tải bộ dữ liệu fer2013 (hoặc có thể tải bộ dữ liệu tiên tiến hơn)
2.Tạo một số thư mục như icons chứa các hình ảnh cảm xúc,captures để lưu ảnh chụp,data để lưu bộ dữ liệu khuôn mặt
3.Dùng hàm main.py để chạy phần mềm:
Code:
import asyncio
import platform
from gui.gui_app import EmotionApp

if __name__ == '__main__':
    app = EmotionApp('emotion_model.h5')
    if platform.system() == "Emscripten":
        asyncio.ensure_future(app.run())
    else:
        asyncio.run(app.run())

![Hình ảnh 1]https://github.com/thongthai3623/face_emotions/blob/main/image/Screenshot%202025-09-25%20183632.png?raw=true





