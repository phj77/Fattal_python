import os
import subprocess

scripts = [
    'fattal_improved_guided_filter.py',
    'fattal_improved_attenuation_map.py',
    'fattal_improved_fft_padding.py'
]

for script in scripts:
    path = os.path.join(r'C:\Users\Park_HyoungJun\LAB\mission\poor_battery_enhancement\src\Fattal_python\src', script)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the input_dir path
    old_code1 = "input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'data_one', '1')"
    new_code1 = "input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'unenhanced_ver')"
    content = content.replace(old_code1, new_code1)
    
    old_fallback = "input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'unenhanced_ver')"
    new_fallback = "input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '1')"
    content = content.replace(old_fallback, new_fallback)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {script}, running...")
    subprocess.run([r'C:\Users\Park_HyoungJun\AppData\Local\Programs\Python\Python312\python.exe', path], cwd=r'C:\Users\Park_HyoungJun\LAB\mission\poor_battery_enhancement\src\Fattal_python\src')

