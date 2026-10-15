import cv2
import sys
import numpy as np

# 1. INTELIGENTNE WCZYTYWANIE ŚCIEŻKI
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "OPENCV_DATA/01/TEST.jpg"

# cv2.imread(filename)
# filename: Ścieżka do pliku na dysku.
img = cv2.imread(img_path)

if img is None:
    print(f"Błąd: Nie znaleziono pliku: {img_path}")
    sys.exit()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# progowanie


retval, dst = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
# src: Input grayscale image (must be single-channel).
# dst: Output image after thresholding (same size and type as input).
# thresh: Threshold value to compare each pixel against.
# maxval: Value assigned to pixels that meet the threshold condition (varies by type).
# type: Thresholding type.
# --- 2. FILTROWANIE ---

# ksize: Rozmiar jądra/maski (szerokość, wysokość). Muszą to być liczby nieparzyste i dodatnie (np. (3,3), (5,5), (9,9)). Im większe, tym silniejsze rozmycie.
# sigmaX: Odchylenie standardowe w osi X. Wpisanie 0 sprawia, że OpenCV wyliczy to sobie samo na podstawie ksize.
low_pass = cv2.GaussianBlur(gray_img, (5, 5), 0)


# ddepth: Głębia wyjściowa obrazu. Używamy cv2.CV_64F (64-bitowe zmiennoprzecinkowe), 
#         ponieważ przejścia z jasnego w ciemne mogą dać wartości ujemne (zwykłe 0-255 by je ucięło).
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)

# cv2.convertScaleAbs(src)
# src: Obraz źródłowy. Wyciąga wartość bezwzględną (usuwa minusy) i konwertuje z powrotem do standardowego formatu 8-bitowego (0-255).
high_pass = cv2.convertScaleAbs(laplacian)

# C) Filtr Sobela (Krawędzie kierunkowe)
# cv2.Sobel(src, ddepth, dx, dy, ksize)
# src: Obraz źródłowy.
# ddepth: Głębia wyjściowa obrazu (podobnie jak w Laplace, używamy cv2.CV_64F żeby nie zgubić danych).
# dx: Rząd pochodnej w osi X (1 = szukaj pionowych krawędzi, 0 = ignoruj).
# dy: Rząd pochodnej w osi Y (1 = szukaj poziomych krawędzi, 0 = ignoruj).
# ksize: Rozmiar maski analizującej (np. 3, 5, 7).
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

# cv2.bitwise_or(src1, src2)
# Logiczne nakładanie na siebie obrazów. Łączy wykryte krawędzie pionowe (sobel_x) i poziome (sobel_y) w jeden spójny obraz.
sobel = cv2.bitwise_or(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))

# D) Detekcja Krawędzi Canny
# cv2.Canny(image, threshold1, threshold2)
# image: Obraz źródłowy.
# threshold1: Próg dolny (tu: 100). Piksele o intensywności poniżej tej wartości na pewno NIE są krawędzią (są odrzucane).
# threshold2: Próg górny (tu: 200). Piksele o intensywności powyżej tej wartości na pewno SĄ krawędzią.
# (Wszystko pomiędzy 100 a 200 jest uznawane za krawędź TYLKO, jeśli dotyka piksela powyżej progu 200).
canny = cv2.Canny(gray_img, 100, 200)

# E) Operacje Morfologiczne (Minimalny / Maksymalny)
# np.ones(shape, dtype)
# Tworzy sztuczną matrycę z samych jedynek.
# shape: Wymiary (5, 5).
# dtype: Typ danych, np.uint8 to standardowe wartości pikseli od 0 do 255.
kernel = np.ones((5, 5), np.uint8)

# cv2.erode(src, kernel, iterations)  <- FILTR MINIMALNY
# cv2.dilate(src, kernel, iterations) <- FILTR MAKSYMALNY
# src: Obraz źródłowy.
# kernel: Matryca (maska), która decyduje, z jak dużego obszaru wyciągamy minimum/maksimum (tu nasza matryca 5x5 z jedynek).
# iterations: Ile razy pod rząd chcemy nałożyć ten efekt na zdjęcie (1 to standard).
filter_min = cv2.erode(gray_img, kernel, iterations=1)
filter_max = cv2.dilate(gray_img, kernel, iterations=1)

# --- 3. WYŚWIETLANIE WYNIKÓW NA ŻYWO ---
# cv2.imshow(winname, mat)
# winname: Tekst, który wyświetli się na pasku tytułowym okienka.
# mat: Zmienna z obrazem (macierz), którą chcemy wyświetlić.
cv2.imshow("1. Oryginal (Szary)", gray_img)
cv2.imshow("2. Low-Pass (Gauss)", low_pass)
cv2.imshow("3. High-Pass (Laplace)", high_pass)
cv2.imshow("Sobel X", cv2.convertScaleAbs(sobel_x))
cv2.imshow("Sobel Y", cv2.convertScaleAbs(sobel_y))
cv2.imshow("4. Sobel", sobel)
cv2.imshow("5. Canny", canny)
cv2.imshow("6. Minimalny (Erozja)", filter_min)
cv2.imshow("7. Maksymalny (Dylatacja)", filter_max)


# cv2.waitKey(delay)
# delay: Czas czekania w milisekundach. Wartość 0 oznacza "zamroź program i czekaj w nieskończoność na dowolny klawisz".
cv2.waitKey(0)

# cv2.destroyAllWindows()
# Brak parametrów. Funkcja sprząta RAM i zamyka wszystkie okna stworzone przez cv2.imshow.
cv2.destroyAllWindows()