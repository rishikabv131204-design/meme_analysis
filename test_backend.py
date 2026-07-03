import sys
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from v import process_video
    print(process_video('oneminute.mp4'))
except Exception as e:
    import traceback
    traceback.print_exc()
