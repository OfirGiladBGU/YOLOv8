import cv2
import numpy as np
from ultralytics.utils.plotting import colors

from collections import defaultdict
import pandas as pd
import math


def read_csv_data(csv_path: str, chunk_size: int = 21):
    df = pd.read_csv(csv_path)
    num_points = len(df)
    num_chunks = num_points // chunk_size
    points_data = []

    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]
        list_x = [np.nanmean(chunk[f'x_{idx}']) for idx in range(len(chunk))]
        list_y = [np.nanmean(chunk[f'y_{idx}']) for idx in range(len(chunk))]

        points_data.append({
            "list_x": list_x,
            "list_y": list_y
        })

    return points_data


def predict(video_path: str, csv1_path: str, csv2_path: str, output_path: str, ids_map: dict):
    csv1_set_of_points = read_csv_data(csv1_path)
    csv2_set_of_points = read_csv_data(csv2_path)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    frame_number = -1
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_number += 1
            track_ids = [0, 1]

            csv1_points = csv1_set_of_points[frame_number]
            csv2_points = csv2_set_of_points[frame_number]

            csv_points_list = [csv1_points, csv2_points]

            # Add Text
            cv2.putText(frame, ids_map[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(0), True), 2)
            cv2.putText(frame, ids_map[1], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(1), True), 2)
            cv2.putText(frame, "COMMON", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(2), True), 2)

            first_track = track_ids[0]
            seen_points = set()

            for csv_points, track_id in zip(csv_points_list, track_ids):
                point_list_x = csv_points['list_x']
                point_list_y = csv_points['list_y']

                for point_x, point_y in zip(point_list_x, point_list_y):
                    if math.isnan(point_x) or math.isnan(point_y):
                        continue

                    # Frame shape (height, width, channels)
                    point_x = int(point_x * frame.shape[1])
                    point_y = int(point_y * frame.shape[0])
                    point_data = [point_x, point_y]

                    if track_ids != first_track and tuple(point_data) in seen_points:
                        color_id = 2
                    else:
                        color_id = int(track_id)
                        seen_points.add(tuple(point_data))

                    # Center
                    cv2.circle(frame, point_data, 5, colors(color_id, True), -1)

                    # frame[point_data[1], point_data[0], track_id] = 255
                    # frame[point_data[1], point_data[0], 2] = 0


            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = "./girls.mp4"
    csv1_path = "./output/asd_all.csv"
    csv2_path = "./output/td_all.csv"
    output_path = "./output_tracking.avi"
    ids_map = {0: "ASD", 1: "TD"}
    predict(video_path=video_path, csv1_path=csv1_path, csv2_path=csv2_path, output_path=output_path, ids_map=ids_map)


if __name__ == '__main__':
    main()
