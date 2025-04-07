import pandas as pd


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_path, index=False)

    print("Данные очищены и сохранены.")
