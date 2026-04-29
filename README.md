# Rozpoznawanie teczowki - Projekt 2 z Biometrii

Repozytorium zawiera finalna wersje projektu rozpoznawania czlowieka na podstawie obrazu teczowki. Implementacja obejmuje pelny pipeline od preprocessingu i segmentacji, przez normalizacje i kodowanie teczowki, az do porownywania kodow i oceny jakosci na zbiorze MMU Iris Dataset.

## Zawartosc repozytorium

- `src/` - kod zrodlowy aplikacji
- `config/` - pliki konfiguracyjne dla przykladowych danych i dla MMU
- `tests/` - testy jednostkowe
- `zdjecia/` - przykladowe obrazy wejsciowe
- `results_mmu/` - dolaczone artefakty walidacji na MMU
- `sprawozdanie/` - zrodla raportu w LaTeX oraz skompilowany `main.pdf`

## Zakres rozwiazania

Pipeline realizuje kolejne etapy:

- binaryzacje i przygotowanie obrazu w skali szarosci
- segmentacje zrenicy
- segmentacje teczowki
- rozwiniecie teczowki do postaci prostokatnej
- podzial na 8 pasow radialnych
- kodowanie z uzyciem filtra Gabora
- porownywanie kodow maskowana odlegloscia Hamminga
- ewaluacje FAR, FRR, EER i identyfikacji leave-one-out

## Uruchomienie

```powershell
python src\iris\main.py --config config\default_config.json check-env
python src\iris\main.py --config config\default_config.json run
python -m unittest discover -s tests -p "test_*.py"
```

Pelna walidacja na MMU:

```powershell
python src\iris\main.py --config config\mmu_dataset_template.json run
```

Konfiguracja MMU zaklada lokalna obecnosc katalogu `MMU-Iris-Database`.

## Wyniki koncowe dla MMU

- liczba klas teczowki: 90
- liczba obrazow: 450
- liczba par genuine: 900
- liczba par impostor: 100125
- EER: `0.2066`
- prog EER: `0.2160`
- accuracy identyfikacji leave-one-out: `0.8578`

## Uwagi

- Pelny katalog `MMU-Iris-Database` nie jest dolaczony do repozytorium.
- W katalogu `results_mmu/` pozostawiono koncowe wyniki potrzebne do udokumentowania dzialania systemu.
- Raport w katalogu `sprawozdanie/` jest zgodny z aktualnymi wynikami uzyskanymi dla finalnej konfiguracji.
