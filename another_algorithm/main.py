import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import pyautogui

from fishing_bot import FishingBot


@dataclass
class FishingConfig:
    # Обновлённые параметры под ваш цвет (BGR формат)
    template_path: str = "template.png"
    target_lower: Tuple[int, int, int] = (24, 141, 140)  # -10% от вашего цвета
    target_upper: Tuple[int, int, int] = (30, 172, 171)  # +10% от вашего цвета
    center_radius: int = 50
    y_offset: int = 145
    match_threshold: float = 0.9


def show_debug_window(region: np.ndarray, max_loc: Tuple[int, int], max_val: float) -> None:
    """Окно с визуализацией процесса"""
    debug_img = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    # Рисуем прямоугольник вокруг найденного совпадения, если порог превышен
    if max_val > 0.8:
        cv2.rectangle(debug_img, max_loc,
                      (max_loc[0] + 10, max_loc[1] + 10),  # Размер прямоугольника можно скорректировать
                      (0, 255, 0), 2)

    # Текст с уровнем совпадения
    cv2.putText(debug_img, f"{max_val:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    cv2.imshow('Fishing Debug', debug_img)
    cv2.waitKey(1)


def casting_float(time_to_holding: float = 0.93):
    """Забрасываем поплавок"""
    pyautogui.mouseDown()
    time.sleep(time_to_holding)
    pyautogui.mouseUp()


def hooking_fish(config: FishingConfig) -> bool:
    """Подсекаем рыбу при появлении восклицательного знака"""
    center_x, center_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2

    # Зона поиска выше головы персонажа
    region = (
        center_x - config.center_radius,
        center_y - config.center_radius - config.y_offset,
        config.center_radius * 2,
        config.center_radius * 2
    )

    # Загрузка шаблона
    template = cv2.imread(config.template_path, cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow('Fishing Debug', cv2.WINDOW_NORMAL)

    try:
        while True:
            screenshot = pyautogui.screenshot(region=region)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Сопоставление шаблона
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Визуализация отладки
            # show_debug_window(img, max_loc, max_val)

            if max_val >= config.match_threshold:
                print("! обнаружен - подсекаем!")
                pyautogui.mouseDown()
                time.sleep(0.1)
                pyautogui.mouseUp()
                return True

    finally:
        cv2.destroyAllWindows()

def start_fishing():
    casting_float()
    time.sleep(2)
    hooking_fish(FishingConfig())


if __name__ == "__main__":
    time.sleep(3)
    casting_float()
    time.sleep(2)
    hooking_fish(FishingConfig())
    bot = FishingBot()
    bot.run()
