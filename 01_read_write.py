import cv2
import sys

# 1. SPRAWDZANIE ARGUMENTÓW Z TERMINALA
# sys.argv to lista wszystkich słów wpisanych w terminalu.
# sys.argv[0] to nazwa skryptu (np. '01_read_write.py')
# sys.argv[1] to nasz pierwszy argument (ścieżka do zdjęcia)



# Jeśli podał, przypisujemy pierwszy argument do zmiennej
img_path = sys.argv[1]

# 2. WCZYTYWANIE OBRAZU
img = cv2.imread(img_path)

if img is None:
    print(f"Błąd: Nie znaleziono pliku: {img_path}")
    sys.exit()

# 3. PRZETWARZANIE I ZAPIS
print(f"Obraz '{img_path}' wczytany pomyślnie. Wymiary: {img.shape}")

cv2.imshow('Podglad obrazu', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()