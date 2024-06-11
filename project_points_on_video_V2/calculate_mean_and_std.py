import numpy as np
import pandas as pd
import math
import pathlib
import csv


def calculate_mean_and_std(csvs_dir: str, output_dir, ids_map):
    csv_paths = list(pathlib.Path(csvs_dir).rglob("*.csv"))
    readers = []
    for csv_path in csv_paths:
        f = open(csv_path, "r")
        reader = csv.reader(f, delimiter=",")
        readers.append(reader)

    for line_idx in range(45000):
        line_idx_data = {
            "ASD": {"x": [], "y": []},
            "TD": {"x": [], "y": []}
        }
        for reader in readers:
            row = next(reader)
            sign = ids_map[row[0]]
            if len(row) == 3:  # (Sign, X, Y)
                x = int(row[1])
                y = int(row[2])
                line_idx_data[sign]["x"].append(x)
                line_idx_data[sign]["y"].append(y)
            elif len(row) == 5:  # (Sign, X1, Y1, X2, Y2)
                x1 = int(row[1])
                y1 = int(row[2])
                x2 = int(row[3])
                y2 = int(row[4])
                line_idx_data[sign]["x"].append(int(np.mean([x1, x2])))
                line_idx_data[sign]["y"].append(int(np.mean([y1, y2])))
            else:
                raise ValueError("Invalid row length")


def main():
    csvs_dir = "./subjects csv"
    output_dir = "./output"
    ids_map = {"1": "ASD", "0": "TD"}
    calculate_mean_and_std(csvs_dir=csvs_dir, output_dir=output_dir, ids_map=ids_map)


if __name__ == '__main__':
    main()
