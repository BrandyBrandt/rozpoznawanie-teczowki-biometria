# Instrukcja: Wrzucenie na GitHub

## Krok 1: Przygotowanie na GitHubie

1. Zaloguj się na [GitHub.com](https://github.com)
2. Kliknij `+` (górny prawy róg) → `New repository`
3. Nazwa: `iris-recognition` (lub inna nazwa)
4. Opis: `Iris Recognition Biometric System - MMU Dataset`
5. Visibility: **Public** (jeśli chcesz udostępnić) lub **Private**
6. **NIE** inicjuj z README, .gitignore, itp. (już masz)
7. Kliknij `Create repository`

## Krok 2: Inicjalizacja Git lokalnie

```bash
cd C:\Users\olobr\iris-recognition-final

# Inicjalizuj git
git init

# Dodaj wszystko
git add .

# Pierwszy commit
git commit -m "Initial commit: iris recognition system with MMU validation"

# Dodaj remote (zamień USERNAME i REPO_NAME)
git remote add origin https://github.com/USERNAME/iris-recognition.git

# Zmień branch na main (jeśli wymagane)
git branch -M main

# Wrzuć na GitHub
git push -u origin main
```

## Krok 3: Weryfikacja

1. Otwórz https://github.com/USERNAME/iris-recognition
2. Powinieneś widzieć:
   - README.md z opisem
   - src/iris/ z kodem
   - tests/ z testami
   - config/ z konfiguracją
   - sprawozdanie/ z PDF
   - POLECENIE.md z wymaganiami

## Co się wrzuca

```
iris-recognition/
├── README.md                    ✓ Instrukcje
├── POLECENIE.md                 ✓ Co było wymagane
├── .gitignore                   ✓ Czego nie wrzucać
├── pyproject.toml               ✓ Konfiguracja pakietu
├── src/iris/
│   ├── pipeline.py              ✓ Główny kod
│   ├── config.py
│   ├── main.py
│   └── ...
├── tests/
│   ├── test_pipeline_utils.py   ✓ Testy (9/9 OK)
│   └── ...
├── config/
│   ├── config.json              ✓ Konfiguracja MMU
│   └── default_config.json
└── sprawozdanie/
    └── main.pdf                 ✓ Raport finalny
```

## Co się NIE wrzuca

```
❌ MMU-Iris-Database/      (za duży - dodaj do .gitignore)
❌ results/                 (wyniki - dodaj do .gitignore)
❌ zdjecia/                 (dane - dodaj do .gitignore)
❌ __pycache__/             (cache - już w .gitignore)
```

## Jak ktoś uruchomi projekt?

```bash
git clone https://github.com/USERNAME/iris-recognition.git
cd iris-recognition
pip install -e .
python -m unittest discover -s tests -p "test_*.py"
```

⚠️ **UWAGA**: Aby faktycznie uruchomić pipeline, będą potrzebne:
- Dataset MMU (skopiować osobno)
- Lub zmienić `config.json` na inny dataset

Ale **kod i testy są kompletne** i działają.

---

**Gotowe!** Po wrzuceniu repo będzie dostępne do sprawdzenia i oceny.
