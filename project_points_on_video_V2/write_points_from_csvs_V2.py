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
        mean_x = chunk['mean_x'].mean()
        mean_y = chunk['mean_y'].mean()
        std_x = chunk['std_x'].mean()
        std_y = chunk['std_y'].mean()

        points_data.append({
            "mean_x": mean_x,
            "mean_y": mean_y,
            "std_x": std_x,
            "std_y": std_y
        })

    return points_data


def predict(video_path: str, csv1_path: str, csv2_path: str, output_path: str, ids_map: dict):
    csv1_set_of_points = read_csv_data(csv1_path)
    csv2_set_of_points = read_csv_data(csv2_path)

    track_mean_history = defaultdict(lambda: [])
    track_std_history = defaultdict(lambda: [])

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

            for csv_points, track_id in zip(csv_points_list, track_ids):
                point_mean_x = csv_points['mean_x']
                point_mean_y = csv_points['mean_y']
                point_std_x = csv_points['std_x']
                point_std_y = csv_points['std_y']

                if math.isnan(point_mean_x):
                    point_mean_x = None
                else:
                    point_mean_x = int(point_mean_x * frame.shape[1])
                    point_std_x = int(point_std_x * frame.shape[1])

                if math.isnan(point_mean_y):
                    point_mean_y = None
                else:
                    point_mean_y = int(point_mean_y * frame.shape[0])
                    point_std_y = int(point_std_y * frame.shape[0])

                # Store tracking history
                track_mean = track_mean_history[track_id]
                track_std = track_std_history[track_id]

                if point_mean_x is not None and point_mean_y is not None:
                    track_mean.append([point_mean_x, point_mean_y])
                    track_std.append([point_std_x, point_std_y])

                if len(track_mean) > 3:
                    track_mean.pop(0)
                    track_std.pop(0)

                # Plot tracks
                if len(track_mean) > 0:
                    track_points = np.array(track_mean, dtype=np.int32).reshape((-1, 1, 2))

                    # Center
                    cv2.circle(frame, (track_mean[-1]), 1, colors(int(track_id), True), -1)

                    # Trail
                    cv2.polylines(frame, [track_points], isClosed=False, color=colors(int(track_id), True), thickness=2)

                    # STD Circle
                    max_std = int(np.mean(track_std[-1]))
                    cv2.circle(frame, (track_mean[-1]), max_std, colors(int(track_id), True), 1)

                    # Add Text
                    cv2.putText(frame, ids_map[track_id], (track_mean[-1][0] + 10, track_mean[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(track_id), True), 2)

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
    csv1_path = "./output/asd_mean_std.csv"
    csv2_path = "./output/td_mean_std.csv"
    output_path = "./output_tracking.avi"
    ids_map = {0: "ASD", 1: "TD"}
    predict(video_path=video_path, csv1_path=csv1_path, csv2_path=csv2_path, output_path=output_path, ids_map=ids_map)


if __name__ == '__main__':
    main()
