# Analiza aktywności fizycznej

## Struktura Projektu
```
.
├── data/
│   ├── health_data.csv         # Dane dotyczące aktywności
│   └── weather_data.csv        # Dane meteorologiczne
├── lib/
│   └── functions.py            # Funkcje pomocnicze
├── .gitignore
├── README.md
├── requirements.txt
├── sf.pdf                      # Prezentacja projektu
└── steps-forecasting.ipynb     # Notebook z projektem
```

## Etapy Projektu

### 1. Zebranie danych
- Zgromadzenie danych o aktywności
- Zgromadzenie danych pogodowych

### 2. Przygotowanie danych
- Przygotowanie danych o aktywności
- Przygotowanie danych pogodowych

### 3. Analiza danych
- Analiza szeregu czasowego
- Analiza zmiennych dodatkowych
  - Zmienne kalendarzowe
  - Zmienne pogodowe

### 4. Modelowanie
- Porównanie modeli SARIMA i Prophet
- Selekcja i dostrajanie modelu
- Prognoza na danych testowych

## Wymagania systemowe
- Python 3.8+
- Wymagane biblioteki w pliku requirements.txt

## Instalacja i uruchomienie
1. Instalacja zależności:
```bash
pip install -r requirements.txt
```

2. Uruchomienie analizy:
```bash
jupyter notebook steps-forecasting.ipynb
```

