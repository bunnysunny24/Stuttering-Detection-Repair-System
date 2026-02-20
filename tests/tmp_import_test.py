import importlib, sys
sys.path.insert(0, r'D:/Bunny/AGNI')
try:
    m = importlib.import_module('Models.enhanced_audio_preprocessor')
    print('IMPORT OK', hasattr(m, 'EnhancedAudioPreprocessor'))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('IMPORT ERROR', e)
