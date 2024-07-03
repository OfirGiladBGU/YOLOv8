import numpy as np
import pandas as pd
import pathlib
import csv
import os


def calculate_mean_and_std(csvs_dir: str, output_dir: str, ids_map: dict):
    asd_data = {"mean_x": [], "mean_y": [], "std_x": [], "std_y": []}
    td_data = {"mean_x": [], "mean_y": [], "std_x": [], "std_y": []}
    width, height = (1280, 1024)

    csv_paths = list(pathlib.Path(csvs_dir).rglob("*.csv"))
    readers = []
    for csv_path in csv_paths:
        f = open(csv_path, "r")
        reader = csv.reader(f, delimiter=",")
        readers.append(reader)

    for line_idx in range(45000):
        print(f"Processing line: {line_idx}")
        # Grab line i data
        line_idx_data = {
            "ASD": {"x": [], "y": []},
            "TD": {"x": [], "y": []}
        }

        for reader in readers:
            row = next(reader)
            curr_child_id = ids_map[row[0]]
            if len(row) == 3:  # (Sign, X, Y)
                x = int(row[1]) if row[1] != "NaN" else -1
                y = int(row[2]) if row[2] != "NaN" else -1

                # Validate x and y
                if x != -1:
                    line_idx_data[curr_child_id]["x"].append(float(x) / float(width))
                # else:
                #     line_idx_data[curr_child_id]["x"].append(np.nan)
                if y != -1:
                    line_idx_data[curr_child_id]["y"].append(float(y) / float(height))
                # else:
                #     line_idx_data[curr_child_id]["y"].append(np.nan)

            elif len(row) == 5:  # (Sign, X1, X2, Y1, Y2)
                x1 = int(row[1]) if row[1] != "NaN" else -1
                x2 = int(row[2]) if row[2] != "NaN" else -1
                y1 = int(row[3]) if row[3] != "NaN" else -1
                y2 = int(row[4]) if row[4] != "NaN" else -1

                # Validate x and y
                x_list = []
                y_list = []
                if x1 != -1:
                    x_list.append(x1)
                if x2 != -1:
                    x_list.append(x2)
                if y1 != -1:
                    y_list.append(y1)
                if y2 != -1:
                    y_list.append(y2)

                if len(x_list) > 0:
                    mean_x = int(np.mean(x_list))
                    line_idx_data[curr_child_id]["x"].append(float(mean_x) / float(width))
                # else:
                #     line_idx_data[curr_child_id]["x"].append(np.nan)
                if len(y_list) > 0:
                    mean_y = int(np.mean(y_list))
                    line_idx_data[curr_child_id]["y"].append(float(mean_y) / float(height))
                # else:
                #     line_idx_data[curr_child_id]["y"].append(np.nan)

            else:
                raise ValueError("Invalid row length")

        # Calculate line i mean and std
        for curr_child_id in list(line_idx_data.keys()):
            x_list = line_idx_data[curr_child_id]["x"]
            y_list = line_idx_data[curr_child_id]["y"]

            # Validate x and y
            if len(x_list) > 0:
                mean_x = np.nanmean(x_list)
                std_x = np.std(x_list)
            else:
                mean_x = "NaN"
                std_x = "NaN"

            if len(y_list) > 0:
                mean_y = np.nanmean(y_list)
                std_y = np.std(y_list)
            else:
                mean_y = "NaN"
                std_y = "NaN"

            # Append results
            if curr_child_id == "ASD":
                asd_data["mean_x"].append(mean_x)
                asd_data["mean_y"].append(mean_y)
                asd_data["std_x"].append(std_x)
                asd_data["std_y"].append(std_y)
            else:
                td_data["mean_x"].append(mean_x)
                td_data["mean_y"].append(mean_y)
                td_data["std_x"].append(std_x)
                td_data["std_y"].append(std_y)

    # Save results
    asd_df = pd.DataFrame(asd_data)
    asd_filepath = os.path.join(output_dir, "asd_mean_std.csv")
    asd_df.to_csv(asd_filepath, index=False)

    td_df = pd.DataFrame(td_data)
    td_filepath = os.path.join(output_dir, "td_mean_std.csv")
    td_df.to_csv(td_filepath, index=False)


def main():
    csvs_dir = "./subjects csv"
    output_dir = "./output"
    ids_map = {"1": "ASD", "0": "TD"}
    calculate_mean_and_std(csvs_dir=csvs_dir, output_dir=output_dir, ids_map=ids_map)


if __name__ == '__main__':
    main()
