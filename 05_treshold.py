import cv2
import sys
import numpy as np
# 1. INTELIGENTNE WCZYTYWANIE ŚCIEŻKI
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "OPENCV_DATA/01/TEST.jpg"

img = cv2.imread(img_path)

if img is None:
    print(f"Błąd: Nie znaleziono pliku: {img_path}")
    sys.exit()

# Progowanie robimy na obrazie w skali szarości
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. METODY PROGOWANIA ---

# A) Zwykłe progowanie (Na sztywno 127 - dla porównania)
ret_normal, thresh_normal = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# B) Metoda Otsu (Inteligenty próg z histogramu)
# Podajemy próg '0', bo i tak zostanie zignorowany.
# Dodajemy flagę '+ cv2.THRESH_OTSU', co włącza automatyczne szukanie.
# Zmienna 'ret_otsu' będzie zawierała wyliczoną liczbę!
ret_otsu, thresh_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# C) Progowanie Adaptacyjne (Średnia wartość z okolicy)
# Parametry: obraz, max_wartość (255), metoda (ADAPTIVE_THRESH_MEAN_C - czyli średnia), 
# typ (BINARY), wielkość bloku (11x11 pikseli), stała odejmowana od średniej (2 - żeby usunąć szum).
thresh_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


# najczęstsciej wystep[ujacy 
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
most_frequent = int(np.argmax(hist))
ret_normal, thresh_most= cv2.threshold(gray_img, most_frequent, 255, cv2.THRESH_BINARY)

# --- 3. WYŚWIETLANIE WYNIKÓW ---
cv2.imshow("1. Oryginal (Szary)", gray_img)
cv2.imshow("2. Progowanie Zwykle (Sztywno 127)", thresh_normal)
cv2.imshow(f"3. Otsu (Automatyczny prog: {ret_otsu})", thresh_otsu)
cv2.imshow("4. Adaptacyjne (Srednia z okolicy)", thresh_adaptive)
cv2.imshow(f"5. Najczęstsza jasność (Prog: {most_frequent})", thresh_most)


cv2.waitKey(0)
cv2.destroyAllWindows()