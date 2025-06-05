import sys
import os
from tbparse import SummaryReader

folder = sys.argv[1]
root_csv_folder = sys.argv[2]

for experiment in os.listdir(folder):
    print(f"Exporting {experiment}...")

    logging_dir = f'{folder}/{experiment}'
    reader = SummaryReader(logging_dir)

    csv_folder = f"{root_csv_folder}/{experiment}"
    try:
        os.makedirs(csv_folder, exist_ok=True)
    except Exception as e:
        print(f"Cannot create directory: {e}")

    reader.scalars.to_csv(f"{csv_folder}/scalars.csv")
    reader.text.to_csv(f"{csv_folder}/text.csv")
