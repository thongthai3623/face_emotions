import PySimpleGUI as sg
import cv2

class Settings:
    def __init__(self, lang='en', performance_mode='normal', resolution=(1280, 720)):
        self.lang = lang
        self.performance_mode = performance_mode
        self.resolution = resolution

    def show_settings(self, _, app):
        layout = [
            [sg.Text(_('Theme'), size=(15, 1)),
             sg.Combo(['DarkBlue3', 'LightGreen', 'SystemDefault'], default_value=sg.theme(), key='-THEME-')],
            [sg.Text(_('Language'), size=(15, 1)), sg.Combo(['en', 'vi'], default_value=app.lang, key='-LANG-')],
            [sg.Text(_('Performance Mode'), size=(15, 1)),
             sg.Combo(['normal', 'light'], default_value=app.performance_mode, key='-PERFORMANCE-')],
            [sg.Text(_('Resolution'), size=(15, 1)),
             sg.Combo(['640x480', '1280x720', '1920x1080'],
                      default_value=f"{app.resolution[0]}x{app.resolution[1]}", key='-RESOLUTION-')],
            [sg.Button(_('Save'), key='-SAVE-'), sg.Button(_('Cancel'), key='-CANCEL-')]
        ]
        window = sg.Window(_('Settings'), layout, modal=True)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, '-CANCEL-'):
                break
            if event == '-SAVE-':
                app.lang = values['-LANG-']
                app.set_language(app.lang)
                app.performance_mode = values['-PERFORMANCE-']
                resolution_str = values['-RESOLUTION-']
                if resolution_str == '640x480':
                    app.resolution = (640, 480)
                elif resolution_str == '1280x720':
                    app.resolution = (1280, 720)
                elif resolution_str == '1920x1080':
                    app.resolution = (1920, 1080)
                if app.video_capture:
                    app.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, app.resolution[0])
                    app.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, app.resolution[1])
                sg.theme(values['-THEME-'])
                window.close()
                break

        window.close()