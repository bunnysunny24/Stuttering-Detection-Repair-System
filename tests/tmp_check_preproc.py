from Models.enhanced_audio_preprocessor import EnhancedAudioPreprocessor
p = EnhancedAudioPreprocessor()
print('SILENCE_WARN_RATIO=', p.silence_warn_ratio)
print('Logger level:', p.logger.level)
print('Error counts keys:', list(p.error_counts.keys()))
