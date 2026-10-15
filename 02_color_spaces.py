import cv2
import sys

# 1. INTELIGENTNE WCZYTYWANIE ŚCIEŻKI
# Jeśli odpalasz z terminala ze ścieżką -> użyj jej.
# Jeśli odpalasz z F5 (debugger) -> użyj domyślnego zdjęcia.
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "OPENCV_DATA/01/TEST.jpg"


img = cv2.imread(img_path)

if img is None:
    print(f"Błąd: Nie znaleziono pliku: {img_path}")
    sys.exit()


# 3. ZMIANA PRZESTRZENI KOLORÓW
# A) Konwersja BGR -> Skala szarości (Grayscale)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# B) Konwersja BGR -> HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



# 4. ZAPISYWANIE WYNIKÓW
# Używamy dopisków _GRAY i _HSV w nazwach plików


cv2.imshow('Gray', gray_img)
cv2.imshow('HSV', hsv_img)

