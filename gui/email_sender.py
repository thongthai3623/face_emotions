import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import PySimpleGUI as sg
import re

class EmailSender:
    def send_email(self, image_path, _):
        layout = [
            [sg.Text(_('Sender Email:')), sg.Input(key='-SENDER_EMAIL-')],
            [sg.Text(_('App Password:')), sg.Input(key='-PASSWORD-', password_char='*')],
            [sg.Text(_('Recipient Email:')), sg.Input(key='-RECEIVER_EMAIL-')],
            [sg.Button(_('Send')), sg.Button(_('Cancel'))]
        ]
        window = sg.Window(_('Send Email'), layout, modal=True)
        sender_email = None
        password = None
        receiver_email = None

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, _('Cancel')):
                window.close()
                return
            if event == _('Send'):
                sender_email = values['-SENDER_EMAIL-'].strip()
                password = values['-PASSWORD-'].strip()
                receiver_email = values['-RECEIVER_EMAIL-'].strip()

                email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
                if not sender_email or not re.match(email_pattern, sender_email):
                    sg.popup_error(_('Invalid sender email! Must be a valid Gmail address.'))
                    continue
                if not receiver_email or not re.match(email_pattern, receiver_email):
                    sg.popup_error(_('Invalid recipient email!'))
                    continue
                if not password or len(password) != 16 or ' ' in password:
                    sg.popup_error(_('Invalid App Password! It should be 16 characters with no spaces.'))
                    continue

                confirm = sg.popup_yes_no(
                    f"Sender Email: {sender_email}\nApp Password: {password}\nRecipient Email: {receiver_email}\n\nIs this information correct?"
                )
                if confirm != 'Yes':
                    continue
                break
        window.close()

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = 'Emotion Recognition Result'
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
        msg.attach(img)
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            server.quit()
            sg.popup(_('Email sent successfully!'))
        except Exception as e:
            sg.popup_error(_('Failed to send email: ') + str(e))