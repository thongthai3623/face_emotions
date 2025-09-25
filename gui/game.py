import random

class EmotionGame:
    def __init__(self, emotions, app):
        self.emotions = emotions
        self.app = app
        self.target_emotion = None

    def start_game(self, _):
        self.target_emotion = random.choice(self.emotions)

    def next_level(self, _):
        self.target_emotion = random.choice(self.emotions)