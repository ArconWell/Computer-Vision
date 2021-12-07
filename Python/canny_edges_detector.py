import cv2 as cv


def detect_canny_edges_from_image(img):
    img = cv.imread(img)
    sf = min(640. / img.shape[1], 480. / img.shape[0])
    img = cv.resize(img, (0, 0), None, sf, sf)
    cv.imshow("original", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv.Canny(gray, 0, 50)
    cv.imshow("edges", edges)
    cv.waitKey()


def detect_canny_edges_from_webcamera():
    cap = cv.VideoCapture(0)
    # Инициализация для захвата с веб-камеры
    while True:
        ok, img = cap.read()  # Загружаем очередной кадр
        if not ok:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Конвертируем цветное изображение в монохромное
        gray = cv.GaussianBlur(gray, (7, 7), 1.5)  # Добравляем размытие
        edges = cv.Canny(gray, 1, 50)  # Детектируем ребра
        cv.imshow("edges", edges)  # Отображаем результат
        if cv.waitKey(10) > 0:  # Ожидаем 30 мс
            break  # Если клавиша нажата, то выход из цикла
