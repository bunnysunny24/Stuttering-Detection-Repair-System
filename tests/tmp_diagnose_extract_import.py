import sys, traceback
sys.path.insert(0, 'Models')
try:
    import extract_features_90plus as m
    print('OK import:', getattr(m, '__file__', m))
except Exception:
    print('IMPORT ERROR')
    traceback.print_exc()
