# exe_best_param.py (HE + Pyramid Levels Combination Fattal)
import sys
from pathlib import Path

# 프로젝트 src 경로를 sys.path에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]  # Fattal_python/
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from experiment.he_pyramid_fattal.exe.exe_best_params import main

if __name__ == "__main__":
    main()
