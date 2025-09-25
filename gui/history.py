import sqlite3
import PySimpleGUI as sg
import os
import datetime

class EmotionHistory:
    def __init__(self, captures_dir):
        self.captures_dir = captures_dir
        self.db_path = 'emotion_history.db'
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (timestamp TEXT, emotion TEXT, image_path TEXT)''')
        conn.commit()
        conn.close()

    def save_emotion(self, emotion, image_path):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO history (timestamp, emotion, image_path) VALUES (?, ?, ?)",
                  (timestamp, emotion, image_path))
        conn.commit()
        conn.close()

    def show_history(self, _, app):
        sg.theme('DarkBlue3')
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT timestamp, emotion, image_path FROM history")
        rows = c.fetchall()
        conn.close()

        history_rows = [
            [sg.Text(f"{row[0]} - {row[1]}", font=('Helvetica', 12), text_color='white', pad=(10, 5)),
             sg.Image(data=app.resize_icon(row[2], (100, 100)), pad=(10, 5)) if os.path.exists(row[2]) else
             sg.Text("Image not found", font=('Helvetica', 12), text_color='red', pad=(10, 5))]
            for row in rows
        ]

        history_layout = [
            [sg.Text(_('Emotion History'), font=('Helvetica', 16, 'bold'), text_color='white',
                     background_color='#1E2A44', justification='center', pad=(10, 10))],
            [sg.Column(history_rows, scrollable=True, vertical_scroll_only=True, size=(780, 400),
                       background_color='#2E3B4E', pad=(10, 10))],
            [sg.Push(), sg.Button(_('Close'), key='-CLOSE-', font=('Helvetica', 12), pad=(10, 10),
                                  button_color=('white', '#4682B4'))]
        ]

        history_window = sg.Window(_('Emotion History'), history_layout, modal=True, resizable=False,
                                  size=(800, 500), location=(300, 300), background_color='#2E3B4E',
                                  element_justification='center')

        while True:
            event, _values = history_window.read()
            if event in (sg.WIN_CLOSED, '-CLOSE-'):
                break
        history_window.close()