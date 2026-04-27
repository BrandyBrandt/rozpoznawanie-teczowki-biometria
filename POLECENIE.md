# Polecenie Projektu 2 - Rozpoznawanie Tęczówki

## Główny cel
Zbudowanie kompletnego systemu biometrycznego do rozpoznawania osób na podstawie obrazu tęczówki.

## Co było wymagane (grade 5.0)

### Faza 1: Przetwarzanie wstępne i segmentacja
1. **Preprocessing**: binaryzacja i przygotowanie obrazu w skali szarości
2. **Segmentacja źrenicy**: wyznaczenie granicy źrenicy
3. **Segmentacja tęczówki**: wyznaczenie granicy zewnętrznej tęczówki

### Faza 2: Normalizacja i kodowanie
4. **Normalizacja**: rozwinięcie tęczówki do prostokąta
5. **Podzial strukturalny**: podział znormalizowanego obrazu na 8 pasów radialnych
6. **Kodowanie tęczówki**: implementacja kodu Daugmana
   - Zastosowanie falki Gabora jako filtru kodującego
   - Binarny kod tęczówki (2 bity na próbkę)

### Faza 3: Porównanie i ewaluacja
7. **Porównanie kodów**: odległość Hamminga z maską wiarygodności
8. **Ewaluacja na MMU**: pełna walidacja na zbiorze MMU Iris Dataset
   - Obliczenie FAR (False Acceptance Rate)
   - Obliczenie FRR (False Rejection Rate)
   - Obliczenie EER (Equal Error Rate)
   - Dokładność identyfikacji Leave-One-Out

## Zbiór danych
**MMU Iris Dataset** (poprawnie w tym projekcie):
- 45 osób
- 2 oczy na osobę (traktowane jako osobne klasy)
- 5 obrazów na każde oko
- **Razem: 450 obrazów, 90 klas**

## Wymagane metryki ewaluacji
- ✓ Genuine pairs: porównania tego samego oka
- ✓ Impostor pairs: porównania różnych osób
- ✓ Średnia odległość genuine vs. impostor
- ✓ EER i próg EER
- ✓ Dokładność identyfikacji (leave-one-out)

## Struktura etapów (Etapy 2-8)
1. Preprocessing i binaryzacja
2. Segmentacja źrenicy
3. Segmentacja tęczówki
4. Normalizacja (rozwinięcie)
5. Agregacja w 8 pasach radialnych
6. Kodowanie falkami Gabora
7. Porównanie kodów (Hamming z kompensacją rotacji)
8. Ewaluacja FAR/FRR/EER oraz identyfikacja

## Ważne uwagi implementacyjne
- Kod Daugmana wymaga **poprawnej interpretacji częstotliwości falki Gabora**
- Normalizacja musi **ograniczać górne i dolne części obrazu** (powieki, rzęsy)
- **Obie oczy tej samej osoby** są osobnymi klasami (nie mają być grupowane)
- Porównanie kodów wymaga **kompensacji przesunięcia kątowego** (rotacja tęczówki)

---

**Projekt realizuje WSZYSTKIE wymagania i ewaluuje na poprawnym zbiorze MMU.**
