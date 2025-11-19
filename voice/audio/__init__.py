"""Speech/Audio related modules."""

from .ASR import SenseVoiceRecognizer
from .TTS import TextToSpeechEngine
from .VAD import AdvancedVoiceActivityDetector, VoiceActivityDetector
from .audio_utils import (
    CrossPlatformAudioManager,
    SimplifiedVoiceRecorder,
    SimplifiedAudioPlayer,
)

__all__ = [
    "SenseVoiceRecognizer",
    "TextToSpeechEngine",
    "AdvancedVoiceActivityDetector",
    "VoiceActivityDetector",
    "CrossPlatformAudioManager",
    "SimplifiedVoiceRecorder",
    "SimplifiedAudioPlayer",
]
