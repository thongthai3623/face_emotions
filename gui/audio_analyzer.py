import librosa
import numpy as np


class AudioAnalyzer:
    def analyze_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path)
            energy = np.mean(librosa.feature.rms(y=y))
            if energy > 0.1:  # Ngưỡng tạm thời
                return 'happy'
            else:
                return 'sad'
        except Exception as e:
            print(f"Error analyzing audio {audio_path}: {e}")
            return 'neutral'
