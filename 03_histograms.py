import cv2
import sys
from matplotlib import pyplot as plt

# 1. INTELIGENTNE WCZYTYWANIE ŚCIEŻKI
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "OPENCV_DATA/01/TEST.jpg"

img = cv2.imread(img_path)


debug == 1


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. OBLICZANIE HISTOGRAMU
# calcHist potrzebuje parametrów w nawiasach kwadratowych (jako listy):
# [gray_img] - nasz obraz
# [0]        - indeks kanału (szary ma tylko jeden kanał: 0)
# None       - maska (brak, analizujemy całe zdjęcie)
# [256]      - ile "słupków" na wykresie (chcemy wszystkie od 0 do 255)
# [0, 256]   - zakres wartości
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

if debug == 1:
    # Postaw tu czerwoną kropkę (Breakpoint) i wciśnij F5!
    # W lewym panelu zobaczysz zmienną 'hist'. 
    # To tablica 256 liczb, mówiąca ile jest pikseli o danej jasności.
    pass

# 4. RYSOWANIE I ZAPISYWANIE WYKRESU
plt.figure()
plt.title("Histogram Jasnosci")
plt.xlabel("Wartosc piksela (0=Czarne, 255=Biale)")
plt.ylabel("Ilosc pikseli na zdjeciu")
plt.plot(hist)
plt.xlim([0, 256])

output_hist_path = "OPENCV_DATA/01/WYKRES_HISTOGRAM.jpg"
plt.savefig(output_hist_path)