* Hướng dẫn sử dụng phần mềm python nhận diện cảm xúc trên khuôn mặt: <br>
1.Tải bộ dữ liệu fer2013 (hoặc có thể tải bộ dữ liệu tiên tiến hơn) <br>
2.Tạo một số thư mục như icons chứa các hình ảnh cảm xúc,captures để lưu ảnh chụp,data để lưu bộ dữ liệu khuôn mặt <br>
3.Tải 2 thư mục core và gui chứa các file python của dự án <br>
4.Dùng hàm main.py để chạy phần mềm: <br>
Code: <br>
import asyncio <br>
import platform <br>
from gui.gui_app import EmotionApp <br>

if __name__ == '__main__': <br>
    app = EmotionApp('emotion_model.h5') <br>
    if platform.system() == "Emscripten": <br>
        asyncio.ensure_future(app.run()) <br>
    else: <br> 
        asyncio.run(app.run()) <br>

![Hình ảnh 1](https://github.com/thongthai3623/face_emotions/blob/main/image/Screenshot%202025-09-25%20183632.png?raw=true)




![Hình ảnh 2](https://github.com/thongthai3623/face_emotions/blob/main/image/Screenshot%202025-09-25%20183657.png?raw=true)












