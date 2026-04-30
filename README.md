# Rozpoznawanie tęczówki - Projekt 2 z Biometrii

Repozytorium zawiera finalną wersję projektu rozpoznawania człowieka na podstawie obrazu tęczówki. Implementacja obejmuje pełny pipeline: od preprocessingu i segmentacji, przez normalizację i kodowanie tęczówki, aż do porównywania kodów i oceny jakości na zbiorze MMU Iris Dataset.

## Zawartość repozytorium

- `src/` - kod źródłowy aplikacji
- `config/` - pliki konfiguracyjne uruchomień
- `results_iris/` - wyniki dla całego datasetu MMU Iris
- `results_mmu_small/` - wyniki dla mniejszego podzbioru MMU Iris (20 przypadków)
- `sprawozdanie/` - skompilowany raport `main.pdf`
- `README.md`, `pyproject.toml` - dokumentacja i konfiguracja projektu

## Zakres rozwiązania

Pipeline realizuje kolejne etapy:

- binaryzację i przygotowanie obrazu w skali szarości
- segmentację źrenicy
- segmentację tęczówki
- rozwinięcie tęczówki do postaci prostokątnej
- podział na 8 pasów radialnych
- kodowanie z użyciem filtra Gabora
- porównywanie kodów maskowaną odległością Hamminga
- ewaluację FAR, FRR, EER oraz identyfikacji leave-one-out

## Uruchomienie

Sprawdzenie konfiguracji:

```powershell
python src\iris\main.py --config config\mmu_dataset_template.json check-env
```

Uruchomienie na pełnym MMU Iris:

```powershell
python src\iris\main.py --config config\mmu_dataset_template.json run
```

Uruchomienie na mniejszym podzbiorze (20 przypadków):

```powershell
python src\iris\main.py --config config\mmu_dataset_small.json run
```

Konfiguracja MMU zakłada lokalną obecność katalogu `MMU-Iris-Database` (nie jest dołączany do repozytorium).

## Wyniki

W katalogu `results_iris/` znajdują się wyniki dla całego datasetu MMU Iris.

W katalogu `results_mmu_small/` znajdują się wyniki dla mniejszego zbioru (20 przypadków) i to na podstawie tego zbioru opisano wyniki w sprawozdaniu.

