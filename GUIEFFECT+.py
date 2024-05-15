import os
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageTk
import pixelsort
import tkinter as tk
from tkinter import filedialog, messagebox
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import numpy as np
import io
import cv2

def apply_pixelate(frame, radius):
    height, width = frame.shape[:2]
    pixelated_frame = np.zeros_like(frame)

    x = np.tile(np.arange(width), height).reshape(height, width)
    y = np.repeat(np.arange(height), width).reshape(height, width)

    mask = (x // radius * radius == x) & (y // radius * radius == y)
    pixelated_frame[mask] = frame[mask]

    return pixelated_frame


def neon_vhs(frame):
    # Конвертируем изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем простое пороговое преобразование
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Инвертируем бинарное изображение
    thresh = cv2.bitwise_not(thresh)

    # Создаем пустое RGB-изображение с тем же размером
    neon_frame = np.zeros_like(frame, dtype=np.uint8)

    # Генерируем случайный неоновый цвет (R, G, B)
    neon_color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))

    # Рисуем неоновые цвета на границах объектов
    neon_frame[thresh == 255] = neon_color

    # Накладываем неоновый эффект на исходное изображение
    neon_frame = cv2.addWeighted(frame, 0.8, neon_frame, 0.2, 0)

    return neon_frame
        

def pixelsort(frame, block_size=8):
    height, width, channels = frame.shape
    sorted_frame = frame.copy()

    block_height = height // block_size
    block_width = width // block_size

    for channel in range(channels):
        channel_frame = frame[:, :, channel]

        for y in range(block_height):
            for x in range(block_width):
                block = channel_frame[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size]
                sorted_block = np.sort(block, axis=None)
                sorted_frame[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size, channel] = sorted_block.reshape(block.shape)

    return sorted_frame


def apply_posterize(frame, levels):
    scale = 255 // (levels - 1)
    return (frame // scale) * scale


def apply_solarize(frame, threshold):
    return np.vectorize(lambda x: 255 - x if x < threshold else x)(frame)
    '''
    Эта функция применяет эффект "соляризации" (solarize) 
    к кадру изображения или видео. Она принимает frame (кадр) 
    и threshold (пороговое значение). Функция использует np.vectorize 
    для применения лямбда-функции  к каждому элементу в frame. 
    Лямбда-функция проверяет, меньше ли значение пикселя 
    threshold. Если да, то она вычитает значение пикселя из 255 (инвертирует его), 
    иначе оставляет его неизменным.
    '''

def apply_unsharp_mask(frame, radius, percent, threshold):
    '''
    Эта функция применяет 
    эффект "резкости" (unsharp mask) 
    к кадру изображения или видео. 
    Она принимает следующие параметры:
    - frame (кадр)
    - radius (радиус размытия 
    для создания маски)
    - percent (процент, 
    на который будет усилена резкость)
    - threshold (пороговое 
    значение для определения краев)
    '''
    blurred = neon_vhs(frame)  # Предполагается, что функция neon_vhs уже определена
    
    height, width = frame.shape[:2]
    mask = np.zeros_like(frame)
    
    x = np.tile(np.arange(width), height).reshape(height, width)
    y = np.repeat(np.arange(height), width).reshape(height, width)
    
    diff = frame - blurred
    mask[np.abs(diff) >= threshold] = percent * diff[np.abs(diff) >= threshold]
    
    return frame + mask
    '''
    возвращает исходный кадр с добавленной маской резкости, 
    создавая эффект повышения резкости.
    '''

def apply_contour(frame):
    '''
    Эта функция применяет эффект "контуров" (contour) к кадру видео. 
    '''
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    '''
   определяет ядро свертки для выделения краев.
    '''
    
    # Разделяем изображение на каналы
    red, green, blue = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    '''
    разделяет кадр на красный, зеленый и синий каналы.
    '''
    
    # Применяем свертку к каждому каналу
    red_contour = np.clip(np.convolve(red.flatten(), kernel.flatten(), mode='same'), 0, 255).reshape(red.shape)
    '''
    применяет свертку с ядром к красному каналу, обрезая результат до диапазона [0, 255]
    и преобразуя его обратно в исходную форму
    '''
    green_contour = np.clip(np.convolve(green.flatten(), kernel.flatten(), mode='same'), 0, 255).reshape(green.shape)
    blue_contour = np.clip(np.convolve(blue.flatten(), kernel.flatten(), mode='same'), 0, 255).reshape(blue.shape)
    '''
    выполняют то же, что и red_contour, но для зеленого и синего каналов соответственно
    '''
    # Применяем пороговое значение
    red_contour = np.vectorize(lambda x: 255 if x > 0 else 0)(red_contour)
    '''
    применяет пороговое значение к красному каналу, 
    делая все пиксели либо черными (0), либо белыми (255), в зависимости от их значения.
    '''
    green_contour = np.vectorize(lambda x: 255 if x > 0 else 0)(green_contour)
    blue_contour = np.vectorize(lambda x: 255 if x > 0 else 0)(blue_contour)
    
    # Объединяем каналы обратно
    contour_frame = np.dstack((red_contour, green_contour, blue_contour))
    '''
    объединяет красный, зеленый и синий каналы обратно в единый кадр.
    '''
    return contour_frame
    '''
    возвращает кадр с примененным эффектом контуров.
    '''

def apply_invert(frame):
    return 255 - frame
    '''
    Эта функция применяет эффект "инвертирования" (инверсии) 
    к кадру видео. Она принимает frame (кадр) 
    и возвращает его инвертированную версию. 
    Инвертирование выполняется путем вычитания 
    значений пикселей из 255 (максимальное значение для каналов RGB).
    '''

def apply_grayscale(frame):
    red = frame[:, :, 0].astype(np.float32)
    green = frame[:, :, 1].astype(np.float32)
    blue = frame[:, :, 2].astype(np.float32)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gray = gray.astype(np.uint8)
    return np.repeat(np.expand_dims(gray, axis=-1), 3, axis=-1)

def apply_effect_video(frame, effect, randomness=None):
    if effect == 'pixelate':
        radius = random.randint(2, 10)
        frame = apply_pixelate(frame, radius)
    elif effect == 'neon_vhs':
        frame = neon_vhs(frame)
    elif effect == 'posterize':
        levels = random.randint(2, 8)
        frame = apply_posterize(frame, levels)
    elif effect == 'solarize':
        threshold = random.randint(128, 255)
        frame = apply_solarize(frame, threshold)
    elif effect == 'unsharp_mask':
        radius = random.randint(2, 10)
        percent = random.randint(100, 200)
        threshold = random.randint(2, 10)
        frame = apply_unsharp_mask(frame, radius, percent, threshold)
    elif effect == 'contour':
        frame = apply_contour(frame)
    elif effect == 'invert':
        frame = apply_invert(frame)
    elif effect == 'grayscale':
        frame = apply_grayscale(frame)
    elif effect == 'pixelsort':
        frame = pixelsort(frame, randomness)
    return frame
    '''
    - 'pixelate': 
    применяет эффект 
    пикселизации с радиусом от 2 до 10.
    
    - 'gaussian_blur': применяет 
    размытие по Гауссу с радиусом 
    от 2 до 10.
    
    - 'posterize': применяет 
    постеризацию с количеством 
    уровней цвета от 2 до 8.
    
    - 'solarize': применяет 
    соляризацию с пороговым 
    значением от 128 до 255.
    
    - 'unsharp_mask': применяет 
    эффект резкости с радиусом 
    от 2 до 10, процентом резкости 
    от 100 до 200 
    и пороговым значением от 2 до 10.
    
    - 'contour': применяет 
    эффект контуров.
    
    - 'invert': применяет 
    инвертирование цветов.
    
    - 'grayscale': 
    применяет преобразование 
    в оттенки серого.
    '''

def apply_effect(effect, image, randomness=None):
    '''
    Эта функция принимает 
    три параметра:
    - effect (строка): 
    название эффекта, 
    который нужно применить 
    к изображению
    
    - image (объект Image 
    из модуля PIL): 
    исходное изображение
    
    - randomness 
    (необязательный параметр): 
    указывает на случайность 
    значений параметров эффекта 
    (если применимо)
    '''
    if effect == 'pixelate':
        image = image.filter(ImageFilter.BoxBlur(random.randint(2, 10)))
        '''
    Если эффект 'pixelate', то применяется фильтр 
    ImageFilter.BoxBlur из модуля PIL. 
    Этот фильтр реализует эффект 
    пикселизации (pixelate) изображения. 
    Радиус размытия (степень пикселизации) 
    выбирается случайным образом из диапазона от 2 до 10.
        '''
    elif effect == 'neon_vhs':
        # Конвертируем изображение PIL в массив NumPy
        np_image = np.array(image)

        # Конвертируем изображение в оттенки серого
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

        # Применяем простое пороговое преобразование
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Инвертируем бинарное изображение
        thresh = cv2.bitwise_not(thresh)

        # Создаем пустое RGB-изображение с тем же размером
        neon_image = np.zeros_like(np_image, dtype=np.uint8)

        # Генерируем случайный неоновый цвет (R, G, B)
        neon_color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))

        # Рисуем неоновые цвета на границах объектов
        neon_image[thresh == 255] = neon_color

        # Накладываем неоновый эффект на исходное изображение
        neon_image = cv2.addWeighted(np_image, 0.8, neon_image, 0.2, 0)

        # Конвертируем массив NumPy обратно в изображение PIL
        image = Image.fromarray(neon_image)
    elif effect == 'posterize':
        image = ImageOps.posterize(image, random.randint(2, 8))
    elif effect == 'solarize':
        image = ImageOps.solarize(image, random.randint(128, 255))
    elif effect == 'unsharp_mask':
        # Преобразование изображения в формат numpy array
        image_array = np.array(image)

        # Применение эффекта neon_vhs (предполагается, что функция neon_vhs уже определена)
        blurred = neon_vhs(image_array)

        height, width, channels = image_array.shape
        mask = np.zeros_like(image_array)

        x = np.tile(np.arange(width), height).reshape(height, width)
        y = np.repeat(np.arange(height), width).reshape(height, width)

        diff = image_array - blurred
        threshold = random.randint(2, 10)
        percent = random.randint(100, 200)
        mask[np.abs(diff) >= threshold] = percent * diff[np.abs(diff) >= threshold]

        # Применение эффекта
        result = image_array + mask

        # Преобразование обратно в формат PIL Image
        image = Image.fromarray(result.astype('uint8'))
    elif effect == 'contour':
        image = image.filter(ImageFilter.CONTOUR)
        '''
    Если эффект 'contour', то применяется фильтр 
    ImageFilter.CONTOUR из модуля PIL. 
    Этот фильтр реализует эффект обнаружения контуров на изображении.
        '''
    elif effect == 'pixelsort':
        pixels = image.load()
        width, height = image.size

        # Случайный порядок сортировки пикселей по горизонтали
        for y in range(height):
            row = [pixels[x, y] for x in range(width)]
            sorted_row = row.copy()
            for x in range(width):
                if random.randint(0, 100) < randomness:
                    random_offset = random.randint(-10, 10)  # Изменяем этот диапазон для большей/меньшей интенсивности
                    new_x = x + random_offset
                    new_x = max(0, min(width - 1, new_x))  # Ограничиваем новую позицию границами изображения
                    sorted_row[x] = row[new_x]
            for x in range(width):
                pixels[x, y] = sorted_row[x]

        # Случайный порядок сортировки пикселей по вертикали
        for x in range(width):
            col = [pixels[x, y] for y in range(height)]
            sorted_col = col.copy()
            for y in range(height):
                if random.randint(0, 100) < randomness:
                    random_offset = random.randint(-10, 10)  # Изменяем этот диапазон для большей/меньшей интенсивности
                    new_y = y + random_offset
                    new_y = max(0, min(height - 1, new_y))  # Ограничиваем новую позицию границами изображения
                    sorted_col[y] = col[new_y]
            for y in range(height):
                pixels[x, y] = sorted_col[y]
    elif effect == 'invert':
        image = ImageOps.invert(image)
    elif effect == 'grayscale':
        image = ImageOps.grayscale(image)
    return image

def open_image():
    global original_image
    '''
    Объявление глобальной переменной original_image. 
    Она будет использоваться для хранения исходного изображения.
    '''
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    '''
    Эта строка открывает диалоговое окно для выбора файла. 
    filedialog.askopenfilename - это функция из модуля tkinter.filedialog, 
    которая позволяет пользователю выбрать файл для открытия. 
    Параметр filetypes указывает, какие типы файлов 
    отображать в диалоговом окне. В данном случае, 
    будут показаны файлы с расширениями .jpg и .png.
    '''
    if file_path:
        '''
    Если пользователь выбрал файл 
    (file_path не пустой), 
    выполняется следующий код:
        1. Внутри блока try 
        происходит попытка 
        открыть выбранный файл 
        с помощью 
        Image.open(file_path) 
        из модуля PIL. 
        Открытое изображение 
        сохраняется 
        в переменную original_image.
        
        2. Если при открытии 
        файла возникла ошибка, 
        она перехватывается 
        в блоке except. 
        Затем функция 
        messagebox.showerror 
        из модуля 
        tkinter.messagebox 
        показывает диалоговое окно 
        с сообщением об ошибке, 
        и функция open_image 
        завершается (return).
        '''
        try:
            original_image = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open the image file: {e}")
            return
        update_preview(original_image)
        '''
        Если изображение было успешно открыто, 
        вызывается функция update_preview(original_image). 
        Эта функция, обновляет предварительный просмотр изображения 
        в графическом интерфейсе приложения
        '''

def update_preview(image):
    global preview_image, current_effect
    '''
    Эта строка объявляет 
    две глобальные переменные: 
    preview_image и current_effect. 
    Они будут использоваться 
    для хранения предварительного просмотра 
    изображения и текущего применяемого эффекта соответственно.
    '''
    preview_image = image.resize((400, 400), resample=Image.Resampling.LANCZOS)
    '''
    В этой строке происходит 
    изменение размера исходного изображения image 
    до размера 400x400 пикселей с использованием метода 
    ресамплинга Image.Resampling.LANCZOS. 
    Результат сохраняется в глобальной переменной 
    preview_image.
    '''
    photo = ImageTk.PhotoImage(preview_image)
    '''
    Здесь создается объект 
    PhotoImage из модуля ImageTk, 
    который является оболочкой 
    для изображения preview_image. 
    Этот объект необходим 
    для отображения изображения 
    в графическом интерфейсе Tkinter.
    '''
    preview_label.configure(image=photo)
    '''
    Эта строка предполагает, 
    что в коде существует виджет Label 
    с именем preview_label, 
    который используется 
    для отображения 
    предварительного просмотра изображения. 
    Метод configure настраивает свойство image 
    этого виджета, присваивая ему объект photo, 
    созданный на предыдущем шаге.
    '''
    preview_label.image = photo
    '''
    Эта строка также связана с механизмом 
    управления ресурсами в Tkinter. 
    Объект PhotoImage должен храниться в переменной, 
    чтобы предотвратить его удаление сборщиком мусора. 
    Присваивание photo к атрибуту image 
    виджета preview_label обеспечивает,
    что изображение не будет удалено, 
    пока существует виджет.
    '''

def apply_random_effect():
    '''
    Эта функция предназначена 
    для применения 
    случайного эффекта к изображению.
    '''
    if original_image is None:
        return
    '''
    Сначала функция проверяет, была ли загружена исходная версия изображения 
    (original_image). Если original_image равно None, 
    то функция завершается, не выполняя никаких действий.
    '''
    effect = random.choice(['pixelate', 'neon_vhs', 'posterize', 'solarize', 'unsharp_mask', 'contour', 'pixelsort', 'invert', 'grayscale'])
    '''
    Здесь функция random.choice() 
    из модуля random используется 
    для выбора случайного эффекта 
    из списка доступных эффектов.
    '''
    randomness = None
    if effect == 'pixelsort':
        randomness = random.randint(1, 100)
    '''
    Если выбранный эффект 
    равен 'pixelsort', 
    то переменной randomness 
    присваивается случайное целое число 
    от 1 до 100. Это значение будет 
    использоваться для управления степенью случайности 
    при применении эффекта сортировки пикселей
    '''
    try:
        image = apply_effect(effect, original_image.copy(), randomness)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply the effect: {e}")
        return
    '''
    Здесь вызывается функция apply_effect(), 
    которая применяет выбранный эффект 
    к копии исходного изображения (original_image.copy()). 
    Параметр randomness передается в эту функцию 
    для управления параметрами эффекта 
    (если применимо). Если при применении эффекта 
    возникает исключение, оно перехватывается 
    в блоке except, и функция messagebox.showerror() 
    из модуля tkinter.messagebox показывает диалоговое окно 
    с сообщением об ошибке. Затем функция 
    apply_random_effect() завершается.
    '''
    update_preview(image)
    '''
    Если эффект был успешно применен, 
    функция update_preview(image) 
    вызывается для обновления 
    предварительного просмотра 
    изображения в графическом интерфейсе.
    '''
    global current_effect
    current_effect = effect
    '''
    Наконец, глобальная переменная 
    current_effect обновляется, 
    чтобы отражать 
    применяемый эффект.
    '''
    
def save_image():
    '''
    Эта функция предназначена 
    для сохранения 
    обработанного изображения в файл.
    '''
    if preview_image is None:
        return
    '''
    Первая строка проверяет, 
    была ли загружена 
    предварительная версия изображения 
    (preview_image). 
    Если preview_image равно None, 
    то функция завершается, 
    не выполняя никаких действий.
    '''
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
    '''
    Эта строка открывает диалоговое окно для сохранения файла 
    с помощью функции filedialog.asksaveasfilename из модуля tkinter.filedialog. 
    Параметр defaultextension=".jpg" устанавливает расширение файла по умолчанию на .jpg. 
    Параметр filetypes определяет типы файлов, которые будут отображаться 
    в диалоговом окне выбора файла. В данном случае, 
    это файлы в форматах JPEG и PNG.
    '''
    if file_path:
        try:
            preview_image.save(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save the image: {e}")
    '''
    Если пользователь выбрал путь для сохранения файла 
    (file_path не пустой), выполняется следующий код:
    
    1. Внутри блока try 
    метод save(file_path) 
    объекта preview_image 
    вызывается для 
    сохранения изображения 
    по указанному пути.
    
    2. Если при сохранении файла 
    возникла ошибка, 
    она перехватывается в блоке except.
    Затем функция messagebox.showerror 
    из модуля tkinter.messagebox 
    показывает диалоговое окно 
    с сообщением об ошибке
    '''

def apply_effect_to_video(current_effect):
    '''
    Эта функция предназначена 
    для применения эффекта к видеофайлу. 
    Она принимает один аргумент 
    current_effect, который 
    представляет собой 
    текущий выбранный эффект.
    '''
    video_window = tk.Toplevel(root)
    video_window.title("Apply Effect to Video")
    '''
    Здесь создается новое окно Toplevel (дочернее окно), 
    которое будет использоваться для интерфейса обработки видео. 
    Это окно получает заголовок "Apply Effect to Video".
    '''

    video_label = tk.Label(video_window, text="Select a video file:")
    video_label.pack()

    video_entry = tk.Entry(video_window)
    video_entry.pack()
    '''
    Создаются два виджета: 
    Label для отображения 
    текста "Select a video file:" 
    и Entry для ввода пути к видеофайлу. 
    
    Оба виджета размещаются 
    в окне video_window 
    с помощью метода pack()
    '''
    
    def browse_video():
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        video_entry.delete(0, tk.END)
        video_entry.insert(0, video_path)

    browse_button = tk.Button(video_window, text="Browse", command=browse_video)
    browse_button.pack()
    '''
    Здесь определяется функция 
    browse_video(), 
    которая открывает диалоговое окно 
    для выбора видеофайла. 
    После выбора файла, 
    его путь вставляется 
    в поле video_entry. 
    Также создается кнопка "Browse", 
    которая вызывает функцию 
    browse_video() при нажатии
    '''

    audio_label = tk.Label(video_window, text="Select an audio file (optional):")
    audio_label.pack()

    audio_entry = tk.Entry(video_window)
    audio_entry.pack()
    '''
    Аналогично, 
    создаются виджеты Label и Entry 
    для выбора аудиофайла (необязательно).
    '''

    def browse_audio():
        audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.ogg")])
        audio_entry.delete(0, tk.END)
        audio_entry.insert(0, audio_path)

    browse_audio_button = tk.Button(video_window, text="Browse", command=browse_audio)
    browse_audio_button.pack()

    def process_video(current_effect):
       '''
    Эта функция предназначена 
    для обработки видеофайла 
    с применением выбранного эффекта 
    current_effect.
       '''
       video_path = video_entry.get()
       if not video_path:
         return
         '''
     Сначала функция получает путь 
     к исходному видеофайлу 
     из виджета video_entry. 
     Если путь не указан, 
     функция завершается.
         '''

       output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
       if not output_path:
         return
         '''
     Затем функция открывает 
     диалоговое окно для выбора пути 
     сохранения обработанного 
     видеофайла. 
     Если пользователь не выбрал путь, 
     функция завершается.
         '''

       try:
         clip = VideoFileClip(video_path)
       except Exception as e:
         messagebox.showerror("Error", f"Failed to open the video file: {e}")
         return
         '''
     Функция пытается открыть исходный видеофайл 
     с помощью VideoFileClip из библиотеки moviepy. 
     Если при открытии файла возникает ошибка, 
     она отображается в диалоговом окне сообщения 
     об ошибке, и функция завершается.
         '''

       audio_path = audio_entry.get()
       if audio_path:
         try:
            audio_clip = AudioFileClip(audio_path)
         except Exception as e:
            messagebox.showerror("Error", f"Failed to open the audio file: {e}")
            return
            '''
    Если был указан путь 
    к аудиофайлу, функция пытается открыть его 
    с помощью AudioFileClip из библиотеки moviepy. 
    Если при открытии файла возникает ошибка, 
    она отображается в диалоговом окне сообщения 
    об ошибке, и функция завершается.
            '''

       randomness = None
       if current_effect == 'pixelsort':
         randomness = random.randint(1, 100)
         '''
     Если выбранный эффект 
     равен 'pixelsort', функция генерирует случайное 
     целое число от 1 до 100 и сохраняет его 
     в переменной randomness. 
     Это число будет использоваться для управления степенью 
     случайности эффекта сортировки пикселей.
         '''

       frames = [apply_effect_video(frame, current_effect, randomness) for frame in clip.iter_frames()]
       '''
       Здесь происходит основная обработка видео. 
       Функция clip.iter_frames() возвращает итератор 
       по кадрам исходного видеофайла. 
       Для каждого кадра вызывается функция apply_effect_video 
       с текущим эффектом current_effect 
       и значением randomness (если применимо). 
       Результаты обработки каждого кадра 
       сохраняются в списке frames.
       '''

       try:
         new_clip = ImageSequenceClip(frames, fps=clip.fps)
         output_clip = new_clip
         if audio_path:
            output_clip = new_clip.set_audio(audio_clip)
         output_clip.write_videofile(output_path, codec='libx264', temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
       except Exception as e:
         messagebox.showerror("Error", f"Failed to process the video: {e}")
         '''
     Наконец, функция 
     создает новый видеоклип new_clip из списка 
     обработанных кадров frames 
     с помощью ImageSequenceClip 
     из библиотеки moviepy. 
     Если был указан аудиофайл, 
     аудиодорожка добавляется 
     к новому видеоклипу 
     с помощью метода set_audio. 
     Затем новый видеоклип 
     сохраняется в файл 
     по выбранному пути output_path 
     с помощью метода write_videofile. 
     Если при этом возникает ошибка, 
     она отображается 
     в диалоговом окне сообщения 
     об ошибке.
         '''
         
    process_button = tk.Button(video_window, text=f"Process Video ({current_effect})", command=lambda: process_video(current_effect))
    process_button.pack()
    '''
    Эта часть кода создает 
    кнопку "Process Video" 
    в окне video_window. 
    Текст на кнопке 
    содержит название 
    текущего эффекта current_effect. 
    При нажатии на кнопку 
    вызывается функция 
    process_video(current_effect) 
    с помощью lambda-функции. 
    Кнопка размещается в окне 
    video_window с помощью метода pack().
    '''


    

root = tk.Tk()
root.title("Image Effect Application")

'''

Здесь создается главное окно приложения 

root с помощью tk.Tk(). 

Окну присваивается заголовок 

"Image Effect Application".

'''

original_image = None
preview_image = None
current_effect = None
'''
Объявляются три переменные: 
original_image, 
preview_image и current_effect. 
Они используются 
для хранения исходного изображения, 
обработанного изображения 
и текущего эффекта соответственно. 
По умолчанию все они 
инициализируются значением None.
'''


open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

'''

кнопка для открытия изображения, 

при нажатии вызывается 

функция open_image.

'''

preview_label = tk.Label(root)
preview_label.pack()

'''

метка для отображения 

обработанного изображения

'''

random_effect_button = tk.Button(root, text="Apply Random Effect", command=apply_random_effect)
random_effect_button.pack()

save_button = tk.Button(root, text="Save Image", command=save_image)
save_button.pack()

video_button = tk.Button(root, text="Apply Effect to Video", command=lambda: apply_effect_to_video(current_effect))
video_button.pack()

root.mainloop()
'''
Эта строка запускает 
главный цикл событий Tkinter, 
что позволяет приложению 
обрабатывать события 
и отображать графический интерфейс.
'''

