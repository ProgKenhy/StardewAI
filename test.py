import time

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
import os


class FishingBot:
    def __init__(self):
        self.model = YOLO('models/fish_detector.pt')
        self.is_fishing = False
        self.last_action = time.time()

    def process_frame(self):
        # 1. Делаем скриншот
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. Детекция объектов
        results = self.model(frame, conf=0.05)

        # 3. Поиск рыбы и зоны
        fish = None
        zone = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_y = (y1 + y2) // 2

                if box.cls == 0:  # zone
                    zone = (y1 + y2) // 2

                elif box.cls == 1:  # fish
                    fish = center_y

        # 4. Управление
        if fish and zone:
            self.control_fishing(fish, zone)

        # 5. Отладка (опционально)
        # debug_img = results[0].plot()
        # cv2.imshow('Fishing Bot Debug', debug_img)
        # cv2.waitKey(1)

    def control_fishing(self, fish_y, zone_y):
        """Управление ЛКМ на основе позиций"""
        if fish_y < zone_y - 15:  # Если рыба выше зоны
            if not self.is_fishing:
                pyautogui.mouseDown()
                self.is_fishing = True
                print("Зажимаем ЛКМ")
        else:
            if self.is_fishing:
                pyautogui.mouseUp()
                self.is_fishing = False
                print("Отпускаем ЛКМ")

    def run(self):
        test_image = cv2.imread('test_fish_image.png')
        results = self.model(test_image, conf=0.009)
        debug_img = results[0].plot()
        cv2.imshow('Test Detection', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for result in results:
            for box in result.boxes:
                print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xyxy}")

        # try:
        #     print("Бот запущен. Нажмите Ctrl+C для остановки")
        #     while True:
        #         self.process_frame()
        #         time.sleep(0.05)  # ~20 FPS
        # except KeyboardInterrupt:
        #     pass
        # finally:
        #     if self.is_fishing:
        #         pyautogui.mouseUp()
        #     cv2.destroyAllWindows()


if __name__ == "__main__":

    bot = FishingBot()

    bot.run()
