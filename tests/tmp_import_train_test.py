import sys
import os
import traceback
import py_compile

MODULE_PATH = r'D:\Bunny\AGNI\Models\train_90plus_final.py'

print('Running py_compile...')
try:
    py_compile.compile(MODULE_PATH, doraise=True)
    print('PY_COMPILE_OK')
except Exception:
    print('PY_COMPILE_FAILED')
    traceback.print_exc()
    sys.exit(2)

# Ensure workspace is on sys.path
sys.path.insert(0, r'D:\Bunny\AGNI')
# Prevent CUDA usage during import
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print('Attempting import...')
try:
    import importlib
    m = importlib.import_module('Models.train_90plus_final')
    print('IMPORT_OK')
except Exception:
    print('IMPORT_FAILED')
    traceback.print_exc()
    sys.exit(3)

print('Done')
