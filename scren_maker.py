import time

import pyautogui

time.sleep(15)

for i in range(36, 101):
    pyautogui.screenshot(f'fishing_dataset/images/train/frame_{i}.png')
    time.sleep(0.3)
