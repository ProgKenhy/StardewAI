import time
import os
import pyautogui


class ScreenshotMaker:
    def __init__(self, capture_width=200, capture_height=800):
        screen_width, screen_height = pyautogui.size()

        self.capture_width = capture_width
        self.capture_height = capture_height
        self.start_x = (screen_width - capture_width - 200) // 2
        self.start_y = (screen_height - capture_height) // 2
        self.end_x = self.start_x + capture_width
        self.end_y = self.start_y + capture_height

        # Создаем папки для сохранения скриншотов
        self.train_path = "fishing_dataset/images/train"
        self.val_path = "fishing_dataset/images/val"
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

        print(f"Регион захвата: {self.start_x}, {self.start_y}, {self.capture_width}, {self.capture_height}")
        print(f"Train скриншоты: {self.train_path}")
        print(f"Val скриншоты: {self.val_path}")

    def capture_screenshots(self, start_index=1, end_index=100, delay=0.3):
        print(f"Начало захвата скриншотов. Всего: {end_index - start_index + 1}")
        time.sleep(15)  # Небольшая задержка для подготовки

        for i in range(start_index, end_index + 1):
            region = (self.start_x, self.start_y, self.capture_width, self.capture_height)
            screenshot = pyautogui.screenshot(region=region)

            # Определяем путь сохранения (каждый пятый кадр -> val)
            if i % 5 == 0:
                save_path = self.val_path
            else:
                save_path = self.train_path

            file_path = os.path.join(save_path, f"frame_{i}.png")
            screenshot.save(file_path)
            print(f"Скриншот сохранен: {file_path}")
            time.sleep(delay)

        print("Захват скриншотов завершён!")


if __name__ == "__main__":
    maker = ScreenshotMaker(capture_width=200, capture_height=800)
    maker.capture_screenshots(start_index=1, end_index=100, delay=0.3)
