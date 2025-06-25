import os

import cv2
import numpy as np
import pyautogui
import time
from ultralytics import YOLO


class FishingBot:
    def __init__(self, capture_width=200, capture_height=800):
        self.model = YOLO('models/fish_detector_2.pt')
        self.is_fishing = False

        screen_width, screen_height = pyautogui.size()

        self.capture_width = capture_width
        self.capture_height = capture_height
        self.start_x = (screen_width - capture_width - 200) // 2
        self.start_y = (screen_height - capture_height) // 2
        self.end_x = self.start_x + capture_width
        self.end_y = self.start_y + capture_height

    def show_debug_window(self, region: np.ndarray) -> None:
        """Окно с визуализацией процесса"""
        debug_img = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        cv2.imshow('Fishing', debug_img)
        cv2.waitKey(1)

    def process_frame(self):
        region = (
            self.start_x,
            self.start_y,
            self.capture_width,
            self.capture_height)

        # 1. Делаем скриншот
        screenshot = pyautogui.screenshot(region=region)

        # debug
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # self.show_debug_window(img)

        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # timestamp = time.strftime("%Y%m%d_%H%M%S")  # Метка времени
        # screenshot_path = f"screenshots/screenshot_{timestamp}.png"
        # os.makedirs("screenshots", exist_ok=True)  # Создаем папку, если её нет
        # cv2.imwrite(screenshot_path, frame)
        # print(f"Скриншот сохранен: {screenshot_path}")

        # 2. Детекция объектов
        results = self.model(frame, conf=0.1)

        # 3. Поиск рыбы и зоны
        fish = None
        zone = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_y = (y1 + y2) // 2

                if box.cls == 0:  # fish
                    fish = center_y
                elif box.cls == 1:  # zone
                    zone = center_y

        # 4. Управление
        if fish and zone:
            self.control_fishing(fish, zone)

        # 5. Отладка (опционально)
        # debug_img = results[0].plot()
        # cv2.imshow('Fishing Bot Debug', debug_img)
        # cv2.waitKey(1)

    def control_fishing(self, fish_y, zone_y):
        """Управление ЛКМ на основе позиций"""
        print(fish_y, zone_y)
        if fish_y < zone_y - 15:  # Если рыба выше зоны
            if not self.is_fishing:
                pyautogui.mouseDown()
                self.is_fishing = True
                print("Рыба выше!")
        else:
            if self.is_fishing:
                pyautogui.mouseUp()
                self.is_fishing = False
                print("Рыба ниже!")

    def run(self):
        try:
            print("Бот запущен. Нажмите Ctrl+C для остановки")
            while True:
                self.process_frame()
                time.sleep(0.02)  # ~20 FPS
        except KeyboardInterrupt:
            pass
        finally:
            if self.is_fishing:
                pyautogui.mouseUp()
            cv2.destroyAllWindows()


