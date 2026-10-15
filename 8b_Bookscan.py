import cv2
import sys
import numpy as np
from pathlib import Path
import pytesseract

def show_scaled(winname, input_image):
    """
    Wyświetla obraz, skalując go tak, aby zmieścił się na ekranie 1920x1080.
    Odpowiednik Twojej funkcji showScaled z C++.
    """
    if input_image is None or input_image.size == 0:
        return

    screen_width = 1920
    screen_height = 1080

    max_width = screen_width * 0.9
    max_height = screen_height * 0.9
    
    scale_x = max_width / input_image.shape[1]  # cols to shape[1] w Pythonie
    scale_y = max_height / input_image.shape[0] # rows to shape[0] w Pythonie
    scale = min(1.0, min(scale_x, scale_y))

    # cv2.resize(src, dsize, fx, fy, interpolation)
    # dsize: docelowy rozmiar (tu ignorujemy - None), bo używamy fx i fy (skalowanie procentowe)
    # interpolation: metoda zmiany rozmiaru. INTER_AREA jest najlepsza do pomniejszania.
    preview = cv2.resize(input_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(winname, preview)
    cv2.waitKey(0)

def detect_document_poly(input_image):
    """
    Szuka 4 rogów kartki papieru na zdjęciu używając konturów i aproksymacji.
    """
    # 1. Binarizacja (Czerń i biel)
    binary_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Używamy Otsu do automatycznego znalezienia idealnego progu odcięcia kartki od tła
    _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 2. Szukanie konturów
    # cv2.findContours(image, mode, method)
    # RETR_LIST: Pobiera wszystkie kontury, nie tworzy hierarchii (kto jest wewnątrz kogo).
    # CHAIN_APPROX_SIMPLE: Kompresuje punkty prostej linii.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 3. Znajdź największy kontur na zdjęciu (zakładamy, że to nasza kartka papieru)
    # cv2.contourArea oblicza pole powierzchni konturu
    document_contour = max(contours, key=cv2.contourArea)

    contour_preview = input_image.copy()

    # 4. Aproksymacja wielokąta (Próba zrobienia z nierównego kształtu równego prostokąta)
    # Obliczamy obwód konturu (True oznacza, że kształt jest zamknięty)
    perimeter = cv2.arcLength(document_contour, True)
    
    # epsilon to maksymalny dystans między oryginalnym kształtem a nowym prostokątem.
    # Ustawienie tego na 2% obwodu (0.02) to standardowa i najlepiej działająca wartość.
    epsilon = 0.02 * perimeter
    
    # cv2.approxPolyDP(curve, epsilon, closed)
    approximated_contour = cv2.approxPolyDP(document_contour, epsilon, True)

    # Rysujemy znaleziony, wygładzony kontur (na czerwono)
    cv2.polylines(contour_preview, [approximated_contour], True, (0, 0, 255), 10)

    # 5. Sortowanie punktów, by zgadzały się z: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    if len(approximated_contour) >= 4:
        # Zmieniamy kształt z (4, 1, 2) na (4, 2) dla łatwiejszych obliczeń
        pts = approximated_contour.reshape(-1, 2)
        
        rect = np.zeros((4, 2), dtype="float32")
        
        # Suma współrzędnych X+Y
        # Najmniejsza suma to róg Lewy-Górny (blisko 0,0)
        # Największa suma to róg Prawy-Dolny (duży X, duży Y)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right
        
        # Różnica współrzędnych X-Y
        # Największa różnica to róg Prawy-Górny (duży X, mały Y)
        # Najmniejsza różnica to róg Lewy-Dolny (mały X, duży Y)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left
        
        # Rysowanie kółek w rogach dla podglądu (tak jak w kodzie C++)
        cv2.circle(contour_preview, tuple(rect[0].astype(int)), 20, (255, 0, 0), -1) # Blue
        cv2.circle(contour_preview, tuple(rect[1].astype(int)), 20, (0, 255, 0), -1) # Green
        cv2.circle(contour_preview, tuple(rect[2].astype(int)), 20, (0, 0, 255), -1) # Red
        cv2.circle(contour_preview, tuple(rect[3].astype(int)), 20, (0, 255, 255), -1) # Yellow
        
        show_scaled("Punkty kontrolne i Kontur", contour_preview)
        return rect
        
    return None

def align_document(input_image, detected_points):
    """
    Bierze wykrzywione punkty dokumentu na zdjęciu i "rozciąga" je na idealnie prostą kartkę A4.
    """
    if detected_points is None:
        return input_image

    # A4 w 300 DPI
    short_side = 2480
    long_side = 3508

    # Punkty docelowe - tak chcemy, żeby wyglądała nasza idealna kartka
    target_points = np.array([
        [0, 0],
        [short_side - 1, 0],
        [short_side - 1, long_side - 1],
        [0, long_side - 1]
    ], dtype="float32")

    # Oblicza matematyczną macierz (transformację) przejścia z krzywych punktów na idealne
    transform = cv2.getPerspectiveTransform(detected_points, target_points)

    # cv2.warpPerspective(src, M, dsize)
    # "Naciąga" oryginalne zdjęcie na nową macierz o zadanym rozmiarze
    output = cv2.warpPerspective(input_image, transform, (short_side, long_side))

    return output

def read_text(input_image):
    """
    Używa Tesseract OCR do wyciągnięcia tekstu. Przystosowano do czytania książek.
    """
    binary_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # 1. ZMIANA: Zamiast Otsu używamy progowania adaptacyjnego.
    # Wyrówna to oświetlenie i usunie czarne cienie z zagięcia grzbietu książki!
    # Używamy dużego bloku (np. 21x21 lub 31x31), żeby algorytm analizował szerszy kontekst tła.
    binary_image = cv2.adaptiveThreshold(
        binary_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    # Opcjonalnie: podgląd obrazu binarnego, żeby upewnić się, że cienie zniknęły
    show_scaled("Przygotowano do OCR (Adaptacyjne)", binary_image)

    print("\n--- Analiza OCR w toku... Proszę czekać... ---")
    
    # 2. ZMIANA: Konfiguracja Page Segmentation Mode (PSM)
    # --psm 3: Domyślny, próbuje automatycznie znaleźć bloki tekstu.
    # --psm 6: Zakłada jeden jednolity blok tekstu (często idealne dla pojedynczej strony książki).
    custom_config = r'--oem 3 --psm 6'
    
    detected_text = pytesseract.image_to_string(binary_image, lang='pol', config=custom_config)

    print("\n--- Odczytany tekst: ---")
    print(detected_text)
    print("------------------------\n")

def test_one_image(image_path):
    print(f"Przetwarzam plik: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Nie mogłem wczytać obrazu: {image_path}")
        return

    # Krok 1: Znajdź rogi kartki
    vertices = detect_document_poly(image)
    
    if vertices is not None:
        # Krok 2: Wyprostuj obrazek
        aligned_document = align_document(image, vertices)
        show_scaled("Wyprostowany Dokument", aligned_document)
        
        # Krok 3: Przeczytaj tekst
        read_text(aligned_document)
    else:
        print("Nie znaleziono kształtu przypominającego dokument!")

def test_whole_dir(dir_path):
    path = Path(dir_path)
    found_any = False
    
    # Przeszukuje rekurencyjnie wszystkie pliki .jpg i .png
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            found_any = True
            test_one_image(str(file_path))
            
    if not found_any:
        print(f"Uwaga: W folderze '{dir_path}' nie znaleziono żadnych plików .jpg ani .png!")

if __name__ == "__main__":
    # Jeśli podano argument w terminalu
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        
        if target_path.is_file():
            test_one_image(str(target_path))
        elif target_path.is_dir():
            test_whole_dir(str(target_path))
        else:
            print(f"Ścieżka {target_path} nie istnieje!")
            sys.exit(1)
    else:
        # Domyślny fallback, jeśli odpalasz przez F5 bez parametrów
        default_path = "OPENCV_DATA/01/DOCUMENT.jpg"
        print(f"Brak argumentów. Próbuję wczytać domyślny plik: {default_path}")
        test_one_image(default_path)
        
    cv2.destroyAllWindows()