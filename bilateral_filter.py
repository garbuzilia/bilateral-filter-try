import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit

@jit(nopython=True)
def custom_clip(x, min_val, max_val):
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    else:
        return x

@jit(nopython=True)
def gaussian(x, sigma):
    x = custom_clip(x, -500, 500)  # Ограничиваем значения x
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(custom_clip(-(x ** 2) / (2 * (sigma ** 2)), -500, 500))

@jit(nopython=True)
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

@jit(nopython=True)
def process_pixel(image, row, col, diameter, sigma_i, sigma_s):
    half_diameter = diameter // 2
    new_pixel = np.zeros(image.shape[2], dtype=image.dtype)

    for channel in range(image.shape[2]):
        wp_total = 0
        filtered_image = 0
        for k in range(-half_diameter, half_diameter + 1):
            for l in range(-half_diameter, half_diameter + 1):
                n_x = row + k
                n_y = col + l
                if n_x < 0 or n_x >= len(image) or n_y < 0 or n_y >= len(image[0]):
                    continue
                diff = int(image[n_x, n_y, channel]) - int(image[row, col, channel])
                diff = custom_clip(diff, -500, 500)
                gi = gaussian(diff, sigma_i)
                gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                wp = gi * gs
                filtered_image += image[n_x, n_y, channel] * wp
                wp_total += wp
        if wp_total > 0:
            filtered_image = filtered_image / wp_total
        new_pixel[channel] = int(np.round(filtered_image))
    return row, col, new_pixel

def print_progress_bar(iteration, total, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    print(f'\rПрогресс: |{bar}| {percent}% Завершено', end='\n')
    if iteration == total:
        print()

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape, dtype=image.dtype)
    print("Начало фильтрации изображения")

    total_rows = len(image)
    processed_rows = 0

    with ThreadPoolExecutor() as executor:
        futures = []
        for row in range(len(image)):
            futures.append(executor.submit(process_row, image, row, diameter, sigma_i, sigma_s))

        for future in as_completed(futures):
            row, new_row = future.result()
            new_image[row] = new_row
            processed_rows += 1
            print_progress_bar(processed_rows, total_rows)

    print("Фильтрация изображения завершена")
    return new_image

@jit(nopython=True)
def process_row(image, row, diameter, sigma_i, sigma_s):
    new_row = np.zeros((image.shape[1], image.shape[2]), dtype=image.dtype)
    half_diameter = diameter // 2

    for col in range(len(image[0])):
        for channel in range(image.shape[2]):
            wp_total = 0
            filtered_image = 0
            for k in range(-half_diameter, half_diameter + 1):
                for l in range(-half_diameter, half_diameter + 1):
                    n_x = row + k
                    n_y = col + l
                    if n_x < 0 or n_x >= len(image) or n_y < 0 or n_y >= len(image[0]):
                        continue
                    diff = int(image[n_x, n_y, channel]) - int(image[row, col, channel])
                    diff = custom_clip(diff, -500, 500)
                    gi = gaussian(diff, sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image += image[n_x, n_y, channel] * wp
                    wp_total += wp
            if wp_total > 0:
                filtered_image = filtered_image / wp_total
            new_row[col, channel] = int(np.round(filtered_image))
    return row, new_row

def calculate_sigma(image, factor=0.5):
    # Рассчитываем среднее и стандартное отклонение интенсивностей пикселей
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    # Устанавливаем sigma_i и sigma_s на основе среднего и стандартного отклонения
    sigma_i = std_intensity * factor
    sigma_s = std_intensity * factor
    return sigma_i, sigma_s

def normalize_contrast(original, modified):
    # Преобразуем изображения в формат float32 для нормализации
    original = original.astype(np.float32)
    modified = modified.astype(np.float32)

    # Рассчитываем среднее и стандартное отклонение для оригинального и модифицированного изображений
    mean_orig, std_orig = cv2.meanStdDev(original)
    mean_mod, std_mod = cv2.meanStdDev(modified)

    # Преобразуем средние значения и стандартные отклонения в одномерные массивы
    mean_orig = mean_orig.flatten()
    std_orig = std_orig.flatten()
    mean_mod = mean_mod.flatten()
    std_mod = std_mod.flatten()

    # Нормализуем контрастность модифицированного изображения
    normalized = (modified - mean_mod) * (std_orig / std_mod) + mean_orig

    # Преобразуем обратно в формат uint8
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized

def multi_level_bilateral_filter(image, diameter, sigma_i, sigma_s, levels=1):
    filtered_image = image.copy()
    for level in range(levels):
        print(f"Уровень {level + 1} из {levels}")
        filtered_image = bilateral_filter(filtered_image, diameter, sigma_i, sigma_s)
    return filtered_image

input_dir = "input"
output_dir = "output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.isfile(output_path):
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            print(f"Изображение загружено: {input_path}")
            if image.shape[2] == 4:  # Если изображение имеет альфа-канал
                bgr = image[:, :, :3]
                alpha = image[:, :, 3]
                sigma_i, sigma_s = calculate_sigma(bgr, factor=0.3)  # Уменьшение значения sigma_i и sigma_s
                filtered_bgr = multi_level_bilateral_filter(bgr, 7, sigma_i, sigma_s)
                filtered_bgr = normalize_contrast(bgr, filtered_bgr)
                filtered_image = np.dstack((filtered_bgr, alpha))
            else:
                sigma_i, sigma_s = calculate_sigma(image, factor=0.3)  # Уменьшение значения sigma_i и sigma_s
                filtered_image = multi_level_bilateral_filter(image, 7, sigma_i, sigma_s)
                filtered_image = normalize_contrast(image, filtered_image)
            cv2.imwrite(output_path, filtered_image)
            print(f"Изображение сохранено: {output_path}")
        else:
            print(f"Не удалось загрузить изображение: {input_path}")
    else:
        print(f"Файл уже существует и не будет обработан: {output_path}")
