import pandas as pd

def clean_excel_file(file_path):
    # Charger le fichier Excel
    xls = pd.ExcelFile(file_path)
    cleaned_sheets = {}

    for sheet_name in xls.sheet_names:
        # Charger chaque feuille
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        # Supprimer les lignes entièrement vides
        df.dropna(how='all', inplace=True)

        # Supprimer les colonnes entièrement vides
        df.dropna(axis=1, how='all', inplace=True)

        # Renommer les colonnes si la première ligne est une en-tête
        if df.iloc[0].isna().sum() <= len(df.columns) / 2:
            df.columns = df.iloc[0]  # Renommer les colonnes
            df = df[1:]  # Supprimer la première ligne qui est maintenant l'en-tête

        cleaned_sheets[sheet_name] = df.reset_index(drop=True)

    return cleaned_sheets


def display_sheets(cleaned_sheets):
    for sheet_name, df in cleaned_sheets.items():
        print(f"\nFeuille: {sheet_name}")
        print(df.head())


# Charger et nettoyer les fichiers Excel
files = [
    'phosphore-lacs-je-f-02.03.03.11.xlsx',
    'environnement-qualité-vie-je-f-02.05.06.xlsx',
    'données-climatiques-je-f-02.03.03.02.xlsx'
]

all_cleaned_data = {file: clean_excel_file(file) for file in files}

# Affichage des résultats
for file, cleaned_sheets in all_cleaned_data.items():
    print(f"\n=== {file} ===")
    display_sheets(cleaned_sheets)
