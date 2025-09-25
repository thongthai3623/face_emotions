import PySimpleGUI as sg
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import datetime
from PIL import Image
import io
import matplotlib.pyplot as plt
import threading
import asyncio
import time

# Sử dụng nhập tương đối
from .history import EmotionHistory
from .email_sender import EmailSender
from .game import EmotionGame
from .settings import Settings
class EmotionApp:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_colors = {
            'angry': (0, 0, 255),    # Đỏ
            'disgust': (0, 255, 0),  # Xanh lá
            'fear': (255, 0, 0),     # Xanh dương
            'happy': (0, 255, 255),  # Vàng
            'sad': (255, 255, 0),    # Cyan
            'surprise': (255, 0, 255), # Tím
            'neutral': (128, 128, 128) # Xám
        }
        self.current_frame = None # Trạng thái khung hình
        self.captures_dir = 'captures'
        os.makedirs(self.captures_dir, exist_ok=True) #Tạo kho ảnh chụp,nếu đã tồn tại dùng exist_ok = True để bỏ qua lỗi)
        self.negative_emotion_timer = 0 # Thời gian cảm xúc tiêu cực
        self.negative_emotion_threshold = 10  # giây
        self.lang = 'vi'  # Mặc định là tiếng Việt
        self.performance_mode = 'normal'
        self.resolution = (1280, 720) # Độ phân giải mặc định
        self.latest_emotion = None
        self.emotion_history = [] # List lưu trữ danh sách cảm xúc
        self.max_emotion_history = 50 # Dùng cho analyse,chỉ lưu 50 lần nhận diện cảm xúc gần nhất

        # Biến trạng thái
        self.game_running = False
        self.game_score = 0
        self.game_time_left = 10
        self.last_game_update = time.time()
        self.analysis_window_open = False
        self.instructions_visible = False
        self.webcam_running = False
        self.webcam_thread = None
        self.webcam_frame = None
        self.webcam_lock = threading.Lock()

        # Placeholder và biểu tượng
        self.placeholder_image = np.zeros((720, 960, 3), dtype=np.uint8)
        self.placeholder_image_bytes = cv2.imencode('.png', self.placeholder_image)[1].tobytes()

        # Placeholder trong suốt 300x300 cho pie chart
        transparent_pie = Image.new('RGBA', (250, 250), (0, 0, 0, 0))
        with io.BytesIO() as output:
            transparent_pie.save(output, format='PNG')
            self.transparent_pie_bytes = output.getvalue()

        # Placeholder trong suốt 100x100 cho emotion icon
        transparent_icon = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        with io.BytesIO() as output:
            transparent_icon.save(output, format='PNG')
            self.transparent_icon_bytes = output.getvalue()

        self.emotion_icons = {}
        for emotion in self.emotions:
            icon_path = f'icons/{emotion}.png'
            self.emotion_icons[emotion] = self.resize_icon(icon_path, (100, 100)) if os.path.exists(icon_path) else self.transparent_icon_bytes

        # Từ điển bản dịch
        self.translations = {
            'en': {
                'Emotion Recognition': 'Emotion Recognition',
                'Emotion:': 'Emotion:',
                'Probabilities:': 'Probabilities:',
                'Load Image': 'Load Image',
                'Load Video': 'Load Video',
                'Webcam': 'Webcam',
                'Stop Webcam': 'Stop Webcam',
                'Capture': 'Capture',
                'Gallery': 'Gallery',
                'History': 'History',
                'Share': 'Share',
                'Settings': 'Settings',
                'Game': 'Game',
                'Analyze': 'Analyze',
                'Exit': 'Exit',
                'Choose an image': 'Choose an image',
                'Image saved as': 'Image saved as',
                'No images in gallery!': 'No images in gallery!',
                'Captured Images': 'Captured Images',
                'Delete': 'Delete',
                'Close': 'Close',
                'Are you sure you want to delete': 'Are you sure you want to delete',
                'Deleted': 'Deleted',
                'Email sent successfully!': 'Email sent successfully!',
                'Failed to send email:': 'Failed to send email:',
                'Please express the emotion:': 'Please express the emotion:',
                'Warning: Negative emotion detected for too long!': 'Warning: Negative emotion detected for too long!',
                'Warning': 'Warning',
                'Sender Email:': 'Sender Email:',
                'App Password': 'App Password',
                'Recipient Email:': 'Recipient Email:',
                'Send': 'Send',
                'Please fill in all fields!': 'Please fill in all fields!',
                'Invalid sender email! Must be a valid Gmail address.': 'Invalid sender email! Must be a valid Gmail address.',
                'Invalid recipient email!': 'Invalid recipient email!',
                'Invalid App Password! It should be 16 characters with no spaces.': 'Invalid App Password! It should be 16 characters with no spaces.',
                'Target Emotion Icon': 'Target Emotion Icon',
                'Game: Express the emotion': 'Game: Express the emotion',
                'Time left:': 'Time left:',
                'Score:': 'Score:',
                'Exit Game': 'Exit Game',
                'Game Over': 'Game Over',
                'Image not found': 'Image not found',
                'Level Up!': 'Level Up!',
                'Try Again!': 'Try Again!',
                'Current Emotion:': 'Current Emotion:',
                'N/A': 'N/A',
                'Emotion Analysis': 'Emotion Analysis',
                'View Counts': 'View Counts',
                'Save to Gallery': 'Save to Gallery',
                'No data available': 'No data available',
                'Chart saved as': 'Chart saved as',
                'Emotion Counts': 'Emotion Counts'
            },
            'vi': {
                'Emotion Recognition': 'Nhận diện cảm xúc',
                'Emotion:': 'Cảm xúc:',
                'Probabilities:': 'Xác suất:',
                'Load Image': 'Tải ảnh',
                'Load Video': 'Tải video',
                'Webcam': 'Máy quay',
                'Stop Webcam': 'Tắt máy quay',
                'Capture': 'Chụp ảnh',
                'Gallery': 'Thư viện',
                'History': 'Lịch sử',
                'Share': 'Chia sẻ',
                'Settings': 'Cài đặt',
                'Game': 'Trò chơi',
                'Analyze': 'Phân tích',
                'Exit': 'Thoát',
                'Choose an image': 'Chọn một ảnh',
                'Image saved as': 'Ảnh đã được lưu dưới tên',
                'No images in gallery!': 'Không có ảnh trong thư viện!',
                'Captured Images': 'Ảnh đã chụp',
                'Delete': 'Xóa',
                'Close': 'Đóng',
                'Are you sure you want to delete': 'Bạn có chắc chắn muốn xóa',
                'Deleted': 'Đã xóa',
                'Email sent successfully!': 'Email đã được gửi thành công!',
                'Failed to send email:': 'Gửi email thất bại:',
                'Please express the emotion:': 'Vui lòng thể hiện cảm xúc:',
                'Warning: Negative emotion detected for too long!': 'Cảnh báo: Cảm xúc tiêu cực được phát hiện quá lâu!',
                'Warning': 'Cảnh báo',
                'Sender Email:': 'Email người gửi:',
                'App Password': 'Mật khẩu ứng dụng',
                'Recipient Email:': 'Email người nhận:',
                'Send': 'Gửi',
                'Please fill in all fields!': 'Vui lòng điền đầy đủ tất cả các trường!',
                'Invalid sender email! Must be a valid Gmail address.': 'Email người gửi không hợp lệ! Phải là địa chỉ Gmail hợp lệ.',
                'Invalid recipient email!': 'Email người nhận không hợp lệ!',
                'Invalid App Password! It should be 16 characters with no spaces.': 'Mật khẩu ứng dụng không hợp lệ! Phải có 16 ký tự và không chứa khoảng trắng.',
                'Target Emotion Icon': 'Biểu tượng cảm xúc mục tiêu',
                'Game: Express the emotion': 'Trò chơi: Thể hiện cảm xúc',
                'Time left:': 'Thời gian còn lại:',
                'Score:': 'Điểm:',
                'Exit Game': 'Thoát trò chơi',
                'Game Over': 'Kết thúc trò chơi',
                'Image not found': 'Không tìm thấy ảnh',
                'Level Up!': 'Lên cấp!',
                'Try Again!': 'Thử lại!',
                'Current Emotion:': 'Cảm xúc hiện tại:',
                'N/A': 'Không có',
                'Emotion Analysis': 'Phân tích cảm xúc',
                'View Counts': 'Xem số lần',
                'Save to Gallery': 'Lưu vào thư viện',
                'No data available': 'Không có dữ liệu',
                'Chart saved as': 'Biểu đồ được lưu dưới dạng',
                'Emotion Counts': 'Số lần cảm xúc'
            }
        }
        self._ = lambda s: self.translations[self.lang].get(s, s) # Dùng để chuyển ngôn ngữ

        # Thiết lập theme
        sg.theme('DarkBlue3')
        screen_width, screen_height = sg.Window.get_screen_size()
        self.webcam_width = int(screen_width * 0.5)
        self.webcam_height = int(self.webcam_width * 0.75)
        icon_size = (100, 100)
        pie_chart_size = (250, 250)

        # Layout chính với cấu trúc cố định hơn
        self.layout = [
            [sg.Text(self._('Emotion Recognition'), font=('Helvetica', 24, 'bold'), justification='center',
                     expand_x=True, pad=(10, 10), key='-TITLE-', text_color='white', background_color='#1E2A44')],
            [sg.Frame('', [
                [sg.Column([
                    [sg.Image(key='-IMAGE-', size=(self.webcam_width, self.webcam_height), background_color='black', pad=(10, 10))]
                ], size=(self.webcam_width + 20, self.webcam_height + 20), element_justification='center', background_color='#2E3B4E'),
                 sg.Frame('', [
                    [sg.Image(key='-EMOTION_ICON-', size=icon_size, pad=(10, 10))],
                    [sg.Image(key='-PIE_CHART-', size=pie_chart_size, pad=(10, 10))],
                    [sg.Text(self._('Emotion:'), key='-EMOTION_TEXT-', font=('Helvetica', 20), text_color='white',
                             pad=(15, 10)),
                     sg.Text('', key='-EMOTION-', font=('Helvetica', 20, 'bold'), text_color='#00E676', pad=(15, 10))],
                    [sg.Text(self._('Probabilities:'), key='-PROBS_TEXT-', font=('Helvetica', 14), text_color='white',
                             pad=(15, 10)),
                     sg.Text('', key='-PROBS-', font=('Helvetica', 14, 'bold'), text_color='#FFCA28', pad=(15, 10))],
                    [sg.Column([
                        [sg.Text('Hướng dẫn:', font=('Helvetica', 14, 'bold'), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Nhận diện:\n  + Ảnh, video: nhấn tải ảnh, video và chọn thư mục chứa ảnh, video cần nhận diện.\n  + Webcam: Bật chế độ webcam để nhận diện trực tiếp trên màn hình.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Chụp ảnh: chụp khung hiển thị ảnh (có thể chụp trực tiếp ngay cả trên webcam và video).', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Thư viện: Nơi lưu trữ ảnh đã chụp và bảng phân tích cảm xúc.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Lịch sử: Hiển thị các cảm xúc nhận diện hiện tại và trước đó.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Chia sẻ: có thể dùng để chia sẻ bức ảnh vừa chụp qua email (người dùng cần email, mật khẩu ứng dụng của bản thân\nvà email của người nhận).', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Cài đặt: Cho phép theme trong các cửa sổ nhỏ, ngôn ngữ, chế độ hiệu suất và cấu hình cho khung hiển thị.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Trò chơi: nhận diện cảm xúc mà trò chơi yêu cầu và nhận điểm, thời gian là 10s nếu hết sẽ thua.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))],
                        [sg.Text('- Phân tích: Hiển thị bảng phân tích cảm xúc nhận diện qua 50 lần gần nhất, người dùng có thể xem các cảm xúc xuất hiện\nbao nhiêu lần và có thể lưu vào thư viện.', font=('Helvetica', 12), text_color='white', background_color='#2E3B4E', pad=(10, 5))]
                    ], key='-INSTRUCTIONS_COLUMN-', scrollable=True, vertical_scroll_only=True, size=(int(self.webcam_width * 1.1), 300), background_color='#2E3B4E', visible=False)]
                 ], pad=(10, 10), background_color='#34495E')]
            ], expand_x=True, background_color='#2E3B4E')],
            [sg.Frame('', [
                [sg.Push(),
                 sg.Button(self._('Load Image'), key='-LOAD_IMAGE-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Load Video'), key='-LOAD_VIDEO-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Webcam'), key='-WEBCAM-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Stop Webcam'), key='-STOP_WEBCAM-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Capture'), key='-CAPTURE-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Gallery'), key='-GALLERY-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('History'), key='-HISTORY-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Share'), key='-SHARE-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Settings'), key='-SETTINGS-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Game'), key='-GAME-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Analyze'), key='-ANALYZE-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#4682B4'), visible=True),
                 sg.Button(self._('Exit'), key='-EXIT-', font=('Helvetica', 12), pad=(8, 8),
                           button_color=('white', '#FF6347'), visible=True),
                 sg.Push()]
            ], expand_x=True, background_color='#2E3B4E', key='-BUTTON_FRAME-')]
        ]

        # Tạo cửa sổ
        self.window = sg.Window(self._('Emotion Recognition'), self.layout, finalize=True, resizable=True,
                                size=(screen_width, screen_height), background_color='#2E3B4E')
        self.window.maximize()

        self.video_capture = None # Là đối tượng video hoặc webcam
        self.history_manager = EmotionHistory(self.captures_dir)
        self.email_sender = EmailSender()
        self.game = EmotionGame(self.emotions, self)
        self.settings = Settings(self.lang, self.performance_mode, self.resolution)

    def set_language(self, lang):
        self.lang = lang
        self.refresh_gui()

    def refresh_gui(self):
        if self.window and self.window.TKroot.winfo_exists():
            # Cập nhật tiêu đề và văn bản
            self.window['-TITLE-'].update(self._('Emotion Recognition'))
            self.window['-EMOTION_TEXT-'].update(value=self._('Emotion:'))
            self.window['-PROBS_TEXT-'].update(value=self._('Probabilities:'))
            self.window.TKroot.title(self._('Emotion Recognition'))

            # Định nghĩa các nút cần cập nhật
            button_keys = {
                '-LOAD_IMAGE-': 'Load Image',
                '-LOAD_VIDEO-': 'Load Video',
                '-WEBCAM-': 'Webcam',
                '-STOP_WEBCAM-': 'Stop Webcam',
                '-CAPTURE-': 'Capture',
                '-GALLERY-': 'Gallery',
                '-HISTORY-': 'History',
                '-SHARE-': 'Share',
                '-SETTINGS-': 'Settings',
                '-GAME-': 'Game',
                '-ANALYZE-': 'Analyze',
                '-EXIT-': 'Exit'
            }

            # Cập nhật văn bản và trạng thái hiển thị của các nút
            for key, text in button_keys.items():
                if key in self.window.AllKeysDict:
                    self.window[key].update(text=self._(text), visible=True)

            # Đảm bảo frame chứa các nút hiển thị
            if '-BUTTON_FRAME-' in self.window.AllKeysDict:
                self.window['-BUTTON_FRAME-'].update(visible=True)

            # Cập nhật trạng thái hiển thị của hướng dẫn
            if '-INSTRUCTIONS_COLUMN-' in self.window.AllKeysDict:
                self.window['-INSTRUCTIONS_COLUMN-'].update(visible=self.instructions_visible)

            self.window.refresh()

    def reset_gui(self):
        if self.window and self.window.TKroot.winfo_exists():
            self.current_frame = None
            if '-IMAGE-' in self.window.AllKeysDict:
                self.window['-IMAGE-'].update(data=self.placeholder_image_bytes)
            if '-EMOTION_ICON-' in self.window.AllKeysDict:
                self.window['-EMOTION_ICON-'].update(data=self.transparent_icon_bytes)
            if '-EMOTION-' in self.window.AllKeysDict:
                self.window['-EMOTION-'].update(self._('N/A'))
            if '-PROBS-' in self.window.AllKeysDict:
                self.window['-PROBS-'].update(self._('N/A'))
            if '-PIE_CHART-' in self.window.AllKeysDict:
                self.window['-PIE_CHART-'].update(data=self.transparent_pie_bytes)
            self.negative_emotion_timer = 0
            self.latest_emotion = None
            self.instructions_visible = False
            if '-INSTRUCTIONS_COLUMN-' in self.window.AllKeysDict:
                self.window['-INSTRUCTIONS_COLUMN-'].update(visible=False)
            self.refresh_gui()

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Chuyển định dạng BGR sang gray(ảnh xám)
        img = cv2.resize(img, (48, 48)) # Đặt lại kích thước thành 48x48 để trùng với mẫu ảnh trong mô hình
        img = img_to_array(img) # Chuyển thành numpy_array để xử lý sâu
        img = img.astype('float32') / 255.0 # Chuẩn hóa pixel thành [0,1]
        img = np.expand_dims(img, axis=0) # Thêm một chiều để tạo tensor
        return img # trả về hình ảnh sau khi xử lý

    def resize_icon(self, icon_path, target_size):
        try:
            with Image.open(icon_path) as img:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    return output.getvalue()
        except Exception:
            return self.transparent_icon_bytes

    def create_pie_chart(self, probs, target_size):
        labels = []
        sizes = []
        for emotion in self.emotions:
            prob_value = float(probs.get(emotion, '0%')[:-1])
            if prob_value > 0:
                labels.append(emotion)
                sizes.append(prob_value)

        if not sizes:
            return None

        colors = [self.emotion_colors.get(label, (128, 128, 128)) for label in labels]
        colors = [f'#{r:02x}{g:02x}{b:02x}' for b, g, r in colors]

        fig, ax = plt.subplots(figsize=(target_size[0] / 100, target_size[1] / 100))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()

    def create_bar_chart(self):
        if not self.emotion_history:
            return None
        emotion_counts = {emotion: self.emotion_history.count(emotion) for emotion in self.emotions}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(emotion_counts.keys(), emotion_counts.values())
        ax.set_xlabel('Cảm xúc')
        ax.set_ylabel('Tần suất')
        ax.set_title('Tần suất cảm xúc (50 lần gần nhất)')
        plt.xticks(rotation=45)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.read()

    def start_webcam(self):
        if not self.webcam_running:
            if not self.video_capture:
                for index in range(5):
                    self.video_capture = cv2.VideoCapture(index)
                    if self.video_capture.isOpened():
                        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        print(f"Webcam mở thành công tại chỉ số {index}")
                        break
                    self.video_capture.release()
                    self.video_capture = None
                if not self.video_capture or not self.video_capture.isOpened():
                    sg.popup_error(self._("Không thể mở webcam! Vui lòng kiểm tra thiết bị hoặc chỉ số webcam."), font=('Helvetica', 12))
                    return
            self.webcam_running = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()

    def stop_webcam(self):
        self.webcam_running = False
        if self.webcam_thread:
            self.webcam_thread.join()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.webcam_frame = None
        if self.window and self.window.TKroot.winfo_exists():
            self.reset_gui()
            self.window.maximize()

    def webcam_loop(self):
        while self.webcam_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                with self.webcam_lock:
                    self.webcam_frame = frame
            time.sleep(0.033)

    def process_frame(self, frame, image_key, window):
        if self.performance_mode == 'light':
            frame = cv2.resize(frame, (480, 360)) # cập nhật lại frame khi người dùng chuyển sang chế độ light
        self.current_frame = frame.copy() # Lưu hình sau khi chụp
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # chuyển frame thành ảnh xám để nhận diện
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60, 60)) # Bộ cascade dùng để phân loại và nhận diện
        emotions_detected = [] # Lưu trữ các cảm xúc phát hiện được
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w] # Cắt ảnh khuôn mặt từ ảnh xám
                face_processed = self.preprocess_image(face) # Đưa vào preprocess để tiền xử lý
                prediction = self.model.predict(face_processed)[0] # Dự đoán cảm xúc và trả về 7 cảm xúc
                emotion = self.emotions[np.argmax(prediction)] # Chọn ra cảm xúc có tỉ lệ cao nhất
                emotions_detected.append(emotion) # Thêm khuôn mặt phát hiện được vào danh sách
                color = self.emotion_colors.get(emotion, (0, 255, 0)) #
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # Vẽ hình chữ nhật lên khuôn mặt
                cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2) # Viết cảm xúc thu được trên hình chữ nhật
            if emotions_detected:
                dominant_emotion = max(set(emotions_detected), key=emotions_detected.count) # Lấy cảm xúc xuất hiện nhiều nhất
                self.emotion_history.append(dominant_emotion) # Thêm vào lịch sử
                if len(self.emotion_history) > self.max_emotion_history:
                    self.emotion_history.pop(0) # xóa lich sử khi quá dài
                self.latest_emotion = dominant_emotion # cập nhật cảm xúc chủ đạo
                if dominant_emotion in ['sad', 'angry', 'fear']:
                    self.negative_emotion_timer += 0.033
                    if self.negative_emotion_timer > self.negative_emotion_threshold:
                        sg.popup(self._('Warning: Negative emotion detected for too long!'), title=self._('Warning'), font=('Helvetica', 12))
                        self.negative_emotion_timer = 0
                else:
                    self.negative_emotion_timer = 0
                probs = {self.emotions[i]: f"{prediction[i] * 100:.1f}%" for i in range(len(self.emotions))} # Tạo từ điển ánh xạ cảm xúc với xác suất (tính bằng %).
                probs_str = ', '.join([f"{k}: {v}" for k, v in probs.items()]) #Chuyển từ điển thành chuỗi
                pie_chart = self.create_pie_chart(probs, (300, 300)) # Tạo biều đồ sau khi đã có đầy đủ các thông tin
                self.update_gui(frame, dominant_emotion, probs_str, self.emotion_icons[dominant_emotion], pie_chart, image_key, window)
            else:
                self.latest_emotion = None
                self.update_gui(frame, self._('N/A'), self._('N/A'), self.transparent_icon_bytes, self.transparent_pie_bytes, image_key, window)
        else:
            self.latest_emotion = None
            self.update_gui(frame, self._('N/A'), self._('N/A'), self.transparent_icon_bytes, self.transparent_pie_bytes, image_key, window)

    def update_gui(self, frame, emotion, probs_str, emotion_icon, pie_chart, image_key, window):
        if window and window.TKroot.winfo_exists():
            window_size = (self.webcam_width, self.webcam_height) if image_key == '-IMAGE-' else window[image_key].get_size() if image_key in window.AllKeysDict else (0, 0)
            if window_size and all(isinstance(dim, int) and dim > 0 for dim in window_size):
                self.current_frame = frame
                img_display = cv2.resize(frame, window_size, interpolation=cv2.INTER_AREA)
                imgbytes = cv2.imencode('.png', img_display)[1].tobytes()
                if image_key in window.AllKeysDict:
                    window[image_key].update(data=imgbytes)
            else:
                if image_key in window.AllKeysDict:
                    window[image_key].update(data=self.placeholder_image_bytes)

            is_game_window = '-GAME_IMAGE-' in window.AllKeysDict
            if is_game_window:
                if '-CURRENT_EMOTION-' in window.AllKeysDict:
                    window['-CURRENT_EMOTION-'].update(emotion)
            else:
                if '-EMOTION_ICON-' in self.window.AllKeysDict:
                    self.window['-EMOTION_ICON-'].update(data=emotion_icon)
                if '-EMOTION-' in self.window.AllKeysDict:
                    emotion_color = self.emotion_colors.get(emotion, (0, 255, 0)) if emotion != self._('N/A') else (0, 230, 118)
                    self.window['-EMOTION-'].update(emotion, text_color=f'#{emotion_color[2]:02x}{emotion_color[1]:02x}{emotion_color[0]:02x}')
                if '-PROBS-' in self.window.AllKeysDict:
                    self.window['-PROBS-'].update(probs_str)
                if '-PIE_CHART-' in self.window.AllKeysDict:
                    self.window['-PIE_CHART-'].update(data=pie_chart if pie_chart else self.transparent_pie_bytes)

    async def run_game(self):
        self.start_webcam()
        if not self.webcam_running:
            return
        self.game_running = True
        self.game_score = 0
        self.game_time_left = 10
        self.last_game_update = time.time()
        self.game.start_game(self._)
        target_icon_data = self.emotion_icons[self.game.target_emotion]

        game_webcam_width = 720
        game_webcam_height = 540
        game_window_width = 1400
        game_window_height = 720
        screen_width, screen_height = sg.Window.get_screen_size()
        window_x = (screen_width - game_window_width) // 2
        window_y = (screen_height - game_window_height) // 2

        game_layout = [
            [sg.Text(self._('Game: Express the emotion'), font=('Helvetica', 20, 'bold'), text_color='#FFD700',
                     background_color='#1E2A44', justification='center', pad=(10, 10), expand_x=True)],
            [sg.Column([
                [sg.Image(key='-GAME_IMAGE-', size=(game_webcam_width, game_webcam_height), background_color='black', pad=(10, 10))]
            ], element_justification='center', background_color='#2E3B4E'),
             sg.Column([
                [sg.Image(data=target_icon_data, key='-TARGET_ICON-', size=(150, 150), pad=(10, 5))],
                [sg.Text(self._('Please express the emotion:'), font=('Helvetica', 16), text_color='white', pad=(10, 5)),
                 sg.Text(self.game.target_emotion, key='-TARGET_EMOTION-', font=('Helvetica', 16, 'bold'),
                         text_color='#00FFFF', pad=(10, 5))],
                [sg.Text(self._('Current Emotion:'), font=('Helvetica', 16), text_color='white', pad=(10, 5)),
                 sg.Text('', key='-CURRENT_EMOTION-', font=('Helvetica', 16, 'bold'), text_color='#FFD700', pad=(10, 5))],
                [sg.Text(self._('Time left:'), font=('Helvetica', 16), text_color='white', pad=(10, 5)),
                 sg.Text(str(self.game_time_left), key='-TIME_LEFT-', font=('Helvetica', 16, 'bold'), text_color='#FF4500',
                         pad=(10, 5))],
                [sg.Text(self._('Score:'), font=('Helvetica', 16), text_color='white', pad=(10, 5)),
                 sg.Text(str(self.game_score), key='-SCORE-', font=('Helvetica', 16, 'bold'), text_color='#32CD32',
                         pad=(10, 5))],
                [sg.Button(self._('Exit Game'), key='-EXIT_GAME-', font=('Helvetica', 14),
                           button_color=('white', '#FF6347'), pad=(10, 10), border_width=2)]
            ], element_justification='center', background_color='#2E3B4E')]
        ]

        game_main_layout = [[sg.Column(game_layout, expand_x=True, expand_y=True)]]
        game_window = sg.Window(self._('Game: Express the emotion'), game_main_layout, finalize=True, resizable=False,
                                background_color='#2E3B4E', element_justification='center',
                                size=(game_window_width, game_window_height), location=(window_x, window_y))

        if '-GAME-' in self.window.AllKeysDict:
            self.window['-GAME-'].update(visible=False)
        self.window.refresh()

        while self.game_running:
            event, values = game_window.read(timeout=100)
            if event in (sg.WIN_CLOSED, '-EXIT_GAME-'):
                self.game_running = False
                break

            current_time = time.time()
            if current_time - self.last_game_update >= 1:
                self.game_time_left -= 1
                self.last_game_update = current_time
                time_color = '#FF4500' if self.game_time_left > 3 else '#FF0000'
                if '-TIME_LEFT-' in game_window.AllKeysDict:
                    game_window['-TIME_LEFT-'].update(value=str(self.game_time_left), text_color=time_color)

                if self.game_time_left <= 0:
                    sg.popup(f"{self._('Game Over')}\n{self._('Score:')} {self.game_score}", title=self._('Game Over'),
                             font=('Helvetica', 12))
                    self.game_running = False
                    break

            if self.webcam_running and self.webcam_frame is not None:
                with self.webcam_lock:
                    frame = self.webcam_frame.copy()
                self.process_frame(frame, '-GAME_IMAGE-', game_window)
                if self.latest_emotion and self.latest_emotion == self.game.target_emotion:
                    self.game_score += 10
                    self.game_time_left = 10
                    sg.popup(self._('Level Up!'), title=self._('Level Up!'), font=('Helvetica', 12))
                    self.game.next_level(self._)
                    target_icon_data = self.emotion_icons[self.game.target_emotion]
                    if '-TARGET_ICON-' in game_window.AllKeysDict:
                        game_window['-TARGET_ICON-'].update(data=target_icon_data)
                    if '-TARGET_EMOTION-' in game_window.AllKeysDict:
                        game_window['-TARGET_EMOTION-'].update(value=self.game.target_emotion)
                    if '-SCORE-' in game_window.AllKeysDict:
                        game_window['-SCORE-'].update(value=str(self.game_score))
                    if '-TIME_LEFT-' in game_window.AllKeysDict:
                        game_window['-TIME_LEFT-'].update(value=str(self.game_time_left), text_color='#FF4500')

        game_window.close()
        self.stop_webcam()
        if '-GAME-' in self.window.AllKeysDict:
            self.window['-GAME-'].update(visible=True)
        self.refresh_gui()

    async def open_analysis_window(self):
        self.analysis_window_open = True
        analysis_layout = [
            [sg.Text(self._('Emotion Analysis'), font=('Helvetica', 20, 'bold'), text_color='#FFD700',
                     background_color='#1E2A44', justification='center', pad=(10, 10), expand_x=True)],
            [sg.Image(key='-CHART-', size=(600, 400), background_color='black', pad=(10, 10))],
            [sg.Text('', key='-MESSAGE-', font=('Helvetica', 12), text_color='white', pad=(10, 5))],
            [sg.Button(self._('View Counts'), key='-VIEW_COUNTS-', font=('Helvetica', 12), pad=(8, 8),
                       button_color=('white', '#4682B4')),
             sg.Button(self._('Save to Gallery'), key='-SAVE_CHART-', font=('Helvetica', 12), pad=(8, 8),
                       button_color=('white', '#4682B4')),
             sg.Button(self._('Exit'), key='-EXIT_ANALYSIS-', font=('Helvetica', 12), pad=(8, 8),
                       button_color=('white', '#FF6347'))]
        ]
        analysis_window = sg.Window(self._('Emotion Analysis'), analysis_layout, modal=True, resizable=False,
                                    background_color='#2E3B4E', element_justification='center',
                                    size=(800, 600), location=(200, 200), finalize=True)

        last_update = time.time()
        update_interval = 1

        while self.analysis_window_open:
            event, values = analysis_window.read(timeout=100)
            if event in (sg.WIN_CLOSED, '-EXIT_ANALYSIS-'):
                self.analysis_window_open = False
                break
            if event == '-VIEW_COUNTS-':
                counts = {emotion: self.emotion_history.count(emotion) for emotion in self.emotions}
                counts_str = '\n'.join([f"{emotion}: {count}" for emotion, count in counts.items()])
                sg.popup(counts_str, title=self._('Emotion Counts'), font=('Helvetica', 12))
            if event == '-SAVE_CHART-':
                chart_data = self.create_bar_chart()
                if chart_data:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'chart_{timestamp}.png'
                    image_path = os.path.join(self.captures_dir, filename)
                    with open(image_path, 'wb') as f:
                        f.write(chart_data)
                    sg.popup(f"{self._('Chart saved as')} {filename}", font=('Helvetica', 12))

            current_time = time.time()
            if current_time - last_update >= update_interval:
                if self.emotion_history:
                    chart_data = self.create_bar_chart()
                    if chart_data and '-CHART-' in analysis_window.AllKeysDict:
                        try:
                            analysis_window['-CHART-'].update(data=chart_data)
                        except Exception as e:
                            print(f"Error updating chart: {e}")
                            analysis_window['-CHART-'].update(data=self.transparent_pie_bytes)
                    if '-MESSAGE-' in analysis_window.AllKeysDict:
                        analysis_window['-MESSAGE-'].update('')
                else:
                    if '-CHART-' in analysis_window.AllKeysDict:
                        analysis_window['-CHART-'].update(data=self.transparent_pie_bytes)
                    if '-MESSAGE-' in analysis_window.AllKeysDict:
                        analysis_window['-MESSAGE-'].update(self._('No data available'))
                last_update = time.time()
                await asyncio.sleep(0)

        analysis_window.close()
        self.analysis_window_open = False
        self.refresh_gui()
        plt.close('all')

    async def open_gallery(self):
        captures = [f for f in os.listdir(self.captures_dir) if f.endswith('.png')]
        if not captures:
            sg.popup(self._('No images in gallery!'), font=('Helvetica', 12))
            return

        gallery_layout = []
        target_size = (400, 400)
        for f in captures:
            image_path = os.path.join(self.captures_dir, f)
            image_data = self.resize_icon(image_path, target_size) if os.path.exists(image_path) else None
            gallery_layout.append([
                sg.Image(data=image_data, key=f'-PREVIEW-{f}', enable_events=True, pad=(10, 10)) if image_data else
                sg.Text(self._('Image not found'), key=f'-PREVIEW-{f}', pad=(10, 10)),
                sg.Button(self._('Delete'), key=f'-DELETE-{f}', font=('Helvetica', 12), pad=(10, 10))
            ])

        layout = [
            [sg.Text(self._('Captured Images'), font=('Helvetica', 16), key='-GALLERY_TITLE-'),
             sg.Push(background_color='#2E3B4E'),
             sg.Button(self._('Close'), key='-CLOSE-', font=('Helvetica', 12))],
            [sg.Column(gallery_layout, scrollable=True, vertical_scroll_only=True, size=(900, 600))]
        ]
        gallery_window = sg.Window(self._('Gallery'), layout, modal=True, resizable=True, size=(950, 700))
        while True:
            event, values = gallery_window.read()
            if event in (sg.WIN_CLOSED, '-CLOSE-'):
                break
            if event.startswith('-DELETE-'):
                file_to_delete = event.split('-DELETE-')[1]
                confirm = sg.popup_yes_no(f"{self._('Are you sure you want to delete')} {file_to_delete}?")
                if confirm == 'Yes':
                    os.remove(os.path.join(self.captures_dir, file_to_delete))
                    sg.popup(f"{self._('Deleted')} {file_to_delete}")
                    gallery_window.close()
                    await self.open_gallery()
                break
        gallery_window.close()
        self.refresh_gui()

    async def run(self):
        last_button_check = time.time()
        check_interval = 5
        while True:
            if not self.analysis_window_open and not self.game_running:
                event, values = self.window.read(timeout=20)
            else:
                event, values = None, None

            current_time = time.time()
            if current_time - last_button_check >= check_interval:
                self.refresh_gui()
                last_button_check = current_time

            if event == sg.WIN_CLOSED or event == '-EXIT-':
                self.stop_webcam()
                break
            if event == '-LOAD_IMAGE-':
                file_path = sg.popup_get_file(self._('Choose an image'), file_types=(('Image Files', '*.jpg *.png'),))
                if file_path:
                    img = cv2.imread(file_path)
                    if img is not None:
                        self.process_frame(img, '-IMAGE-', self.window)
                        if not self.instructions_visible:
                            self.instructions_visible = True
                            if '-INSTRUCTIONS_COLUMN-' in self.window.AllKeysDict:
                                self.window['-INSTRUCTIONS_COLUMN-'].update(visible=True)
            if event == '-LOAD_VIDEO-':
                video_path = sg.popup_get_file(self._('Choose a video'), file_types=(('Video Files', '*.mp4 *.avi'),))
                if video_path:
                    self.stop_webcam()
                    self.video_capture = cv2.VideoCapture(video_path)
                    self.start_webcam()
                    if self.instructions_visible:
                        self.instructions_visible = False
                        if '-INSTRUCTIONS_COLUMN-' in self.window.AllKeysDict:
                            self.window['-INSTRUCTIONS_COLUMN-'].update(visible=False)
            if event == '-WEBCAM-':
                self.start_webcam()
                if self.instructions_visible:
                    self.instructions_visible = False
                    if '-INSTRUCTIONS_COLUMN-' in self.window.AllKeysDict:
                        self.window['-INSTRUCTIONS_COLUMN-'].update(visible=False)
            if event == '-STOP_WEBCAM-':
                self.stop_webcam()
            if event == '-CAPTURE-' and self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'capture_{timestamp}.png'
                image_path = os.path.join(self.captures_dir, filename)
                cv2.imwrite(image_path, self.current_frame)
                if self.latest_emotion:
                    self.history_manager.save_emotion(self.latest_emotion, image_path)
                sg.popup(f"{self._('Image saved as')} {filename}", font=('Helvetica', 12))
            if event == '-GALLERY-':
                await self.open_gallery()
            if event == '-HISTORY-':
                self.history_manager.show_history(self._, self)
            if event == '-SHARE-' and self.current_frame is not None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'capture_{timestamp}.png'
                image_path = os.path.join(self.captures_dir, filename)
                cv2.imwrite(image_path, self.current_frame)
                self.email_sender.send_email(image_path, self._)
            if event == '-SETTINGS-':
                self.settings.show_settings(self._, self)
                self.refresh_gui()
            if event == '-GAME-' and not self.game_running:
                await self.run_game()
            if event == '-ANALYZE-':
                await self.open_analysis_window()

            if self.webcam_running and self.webcam_frame is not None and not self.game_running and not self.analysis_window_open:
                with self.webcam_lock:
                    frame = self.webcam_frame.copy()
                self.process_frame(frame, '-IMAGE-', self.window)

        self.stop_webcam()
        self.window.close()

if __name__ == "__main__":
    app = EmotionApp('emotion_model.h5') # gọi mô hình khi bắt đầu chạy ứng dụng