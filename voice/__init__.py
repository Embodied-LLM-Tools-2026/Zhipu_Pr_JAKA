"""
Voice package regrouped into dedicated subpackages.

Submodules:
- voice.agents: planner, engineer, VLM orchestration.
- voice.control: robot APIs, executors, world model constructs.
- voice.perception: observers, localization helpers.
- voice.audio: ASR/TTS/VAD/audio utilities.
- voice.utils: configuration and dependency helpers.
"""

__all__ = ["agents", "control", "perception", "audio", "utils"]
