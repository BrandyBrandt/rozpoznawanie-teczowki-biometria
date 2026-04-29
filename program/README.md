# Rozpoznawanie tęczówki - Projekt 2

To repozytorium zawiera finalną implementację projektu z biometrii dotyczącego rozpoznawania człowieka na podstawie obrazu tęczówki. Aplikacja realizuje pełny pipeline od przygotowania obrazu i segmentacji, przez normalizację i kodowanie, aż do porównywania kodów i oceny skuteczności na MMU Iris Dataset.

## Struktura

- `src/iris/` - kod źródłowy aplikacji
- `config/default_config.json` - konfiguracja dla przykładowych obrazów
- `config/mmu_dataset_template.json` - konfiguracja dla pełnego zbioru MMU
- `results/` - wyniki i artefakty po uruchomieniu
- `results_mmu/` - końcowe artefakty walidacji na MMU
- `tests/` - testy jednostkowe
- `../sprawozdanie_overleaf/` - źródła sprawozdania w LaTeX

## Zakres działania

Implementacja obejmuje:

- binaryzację i przygotowanie obrazu w skali szarości
- segmentację źrenicy
- segmentację tęczówki
- rozwinięcie tęczówki do prostokąta
- podział na 8 radialnych pasów
- kodowanie z użyciem falki Gabora
- porównywanie kodów maskowaną odległością Hamminga
- ewaluację verification i identification

## Szybki start

```powershell
python src\iris\main.py --config config\default_config.json check-env
python src\iris\main.py --config config\default_config.json run
python -m unittest discover -s tests -p "test_*.py"
```

Pełna walidacja na MMU:

```powershell
python src\iris\main.py --config config\mmu_dataset_template.json run
```

## Uwagi

- Domyślny katalog z obrazami to `zdjecia/`.
- Wyniki segmentacji, normalizacji, kodowania i dopasowania są zapisywane w katalogu `results/`.
- Pełny katalog `MMU-Iris-Database` nie jest dołączony do repozytorium; konfiguracja MMU zakłada jego lokalną obecność obok projektu.
- Komplet sprawozdania znajduje się w `../sprawozdanie_overleaf/`.
