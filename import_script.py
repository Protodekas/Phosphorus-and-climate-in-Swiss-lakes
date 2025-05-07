import pandas as pd


def load_excel(file_path):
    try:
        sheets = pd.ExcelFile(file_path).sheet_names
        data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None


# Chargement des fichiers
files = {
    'phosphore': 'phosphorus-lakes.xlsx',
    'qualite_vie': 'quality-of-life.xlsx',
    'climat': 'weather-data.xlsx'
}

datasets = {name: load_excel(file) for name, file in files.items()}


# Aperçu des fichiers chargés
for name, data in datasets.items():
    if data is not None:
        print(f"\n### Aperçu du fichier '{name}' ###")
        for sheet_name, df in data.items():
            print(f"\nFeuille: {sheet_name}")
            print(df.info())
            print(df.head())
    else:
        print(f"\nImpossible de charger le fichier '{name}'")
