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
└── steps-forecasting.ipynb     # Notebook z analizą
```

## Etapy Projektu

### 1. Zebranie danych
- Gromadzenie danych o aktywności
- Gromadzenie danych pogodowych

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
- Walidacja krzyżowa
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
