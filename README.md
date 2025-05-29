### Удаление шумов и восстановление изображений с помощью методов фильтрации

## Функции добавления шума гаусса и импульсного шумма
```Python
def add_gaussian_noise(image, mean=0, sigma=25): 
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
    return noisy_image
```

## Функции для метода Гаусса и быстрого размытия по Гауссу

```Python
def gaussian_kernel(size, sigma=1):
    #Создает ядро Гаусса заданного размера и сигмы.
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_blur_rgb(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = np.zeros_like(image, dtype=float)

    for channel in range(3):  # Предполагаем, что изображение имеет 3 канала (RGB)
        image_padded = np.pad(image[..., channel], kernel_size//2, mode='reflect')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                neighborhood = image_padded[i:i+kernel_size, j:j+kernel_size]
                blurred_image[i, j, channel] = np.sum(neighborhood * kernel)

    return np.clip(blurred_image, 0, 255).astype(np.uint8)
```

```Python
def apply_fast_gaussian_blur_rgb(image, kernel_size=5, sigma=1):
    """Быстрое гауссово размытие через разделение ядра на два одномерных фильтра"""
    kernel = gaussian_kernel(kernel_size, sigma)
    result = np.zeros_like(image, dtype=np.float64)
    
    # Добавляем отступы для обработки границ
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Горизонтальная свертка
    temp = np.zeros_like(padded, dtype=np.float64)
    for channel in range(3):
        for i in range(padded.shape[0]):
            for j in range(pad, padded.shape[1] - pad):
                neighborhood = padded[i, j - pad:j + pad + 1, channel]
                temp[i, j, channel] = np.sum(neighborhood * kernel)
    
    # Вертикальная свертка
    for channel in range(3):
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):  
                neighborhood = temp[i - pad:i + pad + 1, j, channel]
                result[i - pad, j - pad, channel] = np.sum(neighborhood * kernel) 
    
    return np.clip(result, 0, 255).astype(np.uint8)
```
## Средний(усредняющий) фильтр
```Python
def apply_average_filter_rgb(image, kernel_size=5):
    average_filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Получаем половину размера ядра
    half_kernel = kernel_size // 2

    for channel in range(3): 
        for i in range(half_kernel, image.shape[0] - half_kernel):
            for j in range(half_kernel, image.shape[1] - half_kernel):
                # Извлекаем окрестность пикселя
                neighborhood = image[i - half_kernel:i + half_kernel + 1,
                                     j - half_kernel:j + half_kernel + 1,
                                     channel]
                # Вычисляем среднее значение и присваиваем его центральному пикселю
                average_filtered_image[i, j, channel] = np.mean(neighborhood)

    return average_filtered_image
```
## Медианный фильтр
```Python
def apply_median_filter_rgb(image, kernel_size=5):
    median_filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Получаем половину размера ядра
    half_kernel = kernel_size // 2

    for channel in range(3): 
        for i in range(half_kernel, image.shape[0] - half_kernel):
            for j in range(half_kernel, image.shape[1] - half_kernel):
                # Извлекаем окрестность пикселя
                neighborhood = image[i - half_kernel:i + half_kernel + 1,
                                     j - half_kernel:j + half_kernel + 1,
                                     channel]

                # Вычисляем медиану и присваиваем её центральному пикселю
                median_filtered_image[i, j, channel] = np.median(neighborhood)

    return median_filtered_image
```
## Взвешенный медианный фильтр
```Python
def apply_weighted_median_filter_rgb(image, weights=None, kernel_size=5):
    """
    Применяет взвешенный медианный фильтр к каждому каналу изображения.
    Веса определяют, сколько раз пиксель учитывается при вычислении медианы.
    """
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    half_kernel = kernel_size // 2

    # Если веса не заданы, используем однородные веса (эквивалент обычному медианному фильтру)
    if weights is None:
        weights = np.ones((kernel_size, kernel_size), dtype=int)
    else:
        weights = np.array(weights, dtype=int)

    for channel in range(3):
        for i in range(half_kernel, image.shape[0] - half_kernel):
            for j in range(half_kernel, image.shape[1] - half_kernel):
                # Извлекаем окрестность и соответствующие веса
                neighborhood = image[i - half_kernel:i + half_kernel + 1,
                                     j - half_kernel:j + half_kernel + 1,
                                     channel]
                current_weights = weights[:neighborhood.shape[0], :neighborhood.shape[1]]

                # Создаем взвешенный список значений
                weighted_values = []
                for val, weight in zip(neighborhood.flatten(), current_weights.flatten()):
                    weighted_values.extend([val] * weight)  # Дублируем значение по весу

                # Вычисляем медиану
                filtered_image[i, j, channel] = np.median(weighted_values)

    return filtered_image

# Гауссовые весы
gaussian_weights = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]
```
## Адпативный медианный фильтр
```Python 
def apply_adaptive_median_filter_rgb(image, max_kernel_size=7):
    """
    Применяет адаптивный медианный фильтр к цветному изображению (RGB).
    
    :param image: Входное изображение в формате numpy.ndarray.
    :param max_kernel_size: Максимально допустимый размер ядра.
    :return: Отфильтрованное изображение.
    """
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    for channel in range(3):  # Обрабатываем каждый канал отдельно
        padded_channel = np.pad(image[..., channel], max_kernel_size // 2, mode='edge')

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                kernel_size = 3  # Начинаем с минимального окна

                while kernel_size <= max_kernel_size:
                    half_k = kernel_size // 2
                    i_pad = i + max_kernel_size // 2
                    j_pad = j + max_kernel_size // 2

                    # Извлекаем окрестность пикселя
                    neighborhood = padded_channel[i_pad - half_k:i_pad + half_k + 1,
                                                  j_pad - half_k:j_pad + half_k + 1]

                    Z_min = np.min(neighborhood)
                    Z_max = np.max(neighborhood)
                    Z_med = np.median(neighborhood)
                    Z_xy = padded_channel[i_pad, j_pad]

                    A1 = Z_med - Z_min
                    A2 = Z_med - Z_max

                    if A1 > 0 and A2 < 0:
                        # Уровень A: медиана — хороший кандидат
                        B1 = Z_xy - Z_min
                        B2 = Z_xy - Z_max
                        if B1 > 0 and B2 < 0:
                            # Пиксель не шумовой
                            output = Z_xy
                        else:
                            # Пиксель шумовой — заменяем на медиану
                            output = Z_med
                        break
                    else:
                        # Увеличиваем размер окна
                        kernel_size += 2
                else:
                    # Если максимальный размер достигнут — используем медиану
                    output = Z_med

                filtered_image[i, j, channel] = output

    return filtered_image
```

## Билатеральный фильтр
```Python
def apply_bilateral_filter_rgb(image, kernel_size=5, sigma_spatial=10.0, sigma_range=25.0):
    if image.dtype != np.uint8:
        raise ValueError("Ожидается изображение типа uint8.")
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным.")
    
    # Преобразуем в float
    image = image.astype(np.float32)

    # Половина ядра
    half_k = kernel_size // 2

    filtered_image = np.zeros_like(image)

    # Создаем координатную сетку ядра
    x, y = np.meshgrid(np.arange(-half_k, half_k + 1), np.arange(-half_k, half_k + 1))
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    # Применяем фильтр к каждому каналу
    for c in range(3):
        for i in range(half_k, image.shape[0] - half_k):
            for j in range(half_k, image.shape[1] - half_k):
                # Окно вокруг пикселя
                region = image[i - half_k:i + half_k + 1, j - half_k:j + half_k + 1, c]

                # Цветовые (range) веса
                center_val = image[i, j, c]
                range_weights = np.exp(-((region - center_val) ** 2) / (2 * sigma_range**2))

                # Общие билатеральные веса
                weights = spatial_weights * range_weights

                # Нормализация
                weights_sum = np.sum(weights)
                filtered_pixel = np.sum(region * weights) / weights_sum

                filtered_image[i, j, c] = filtered_pixel

    # Приводим обратно к uint8
    return np.clip(filtered_image, 0, 255).astype(np.uint8)
```
NLM фильтр(с использованием OpenCV)
```Python
def apply_non_local_means_rgb(image, h=10, template_window_size=7, search_window_size=21):
    if image.dtype != np.uint8:
        raise ValueError("Ожидается изображение типа uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ожидается RGB-изображение.")
    
    # BGR opencv в RGB
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Применяем встроенную реализацию OpenCV
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h,                  # параметр фильтрации для цветовых каналов
        hColor=h,           # параметр фильтрации для яркости
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )

    # Возвращаем обратно в RGB
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    return denoised_rgb
```
