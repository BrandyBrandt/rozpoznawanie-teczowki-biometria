# MMU Iris Dataset - status walidacji

Pelna walidacja na MMU Iris Dataset zostala wykonana lokalnie.

## Co zostalo sprawdzone

- zbior `MMU-Iris-Database` zostal rozpakowany i uzyty w calym pipeline,
- kod poprawnie obsluguje zagniezdzona strukture katalogow,
- lewe i prawe oko tej samej osoby sa traktowane jako osobne klasy,
- testy jednostkowe przechodza `9/9`,
- pipeline przechodzi etapy `2-8` bez bledow.

## Skala walidacji

- 45 osob,
- 2 oczy na osobe,
- 5 obrazow dla kazdego oka,
- lacznie 450 obrazow i 90 klas teczowki.

## Wyniki finalnej konfiguracji

- `genuine_pairs = 900`
- `impostor_pairs = 100125`
- `genuine_mean_distance = 0.1781`
- `impostor_mean_distance = 0.2272`
- `EER = 0.2066`
- `eer_threshold = 0.2160`
- `identification_accuracy_loo = 0.8578`

## Jak uruchomic ponownie

```powershell
python src/iris/main.py --config config/mmu_dataset_template.json run
python -m unittest discover -s tests -p "test_*.py"
```
