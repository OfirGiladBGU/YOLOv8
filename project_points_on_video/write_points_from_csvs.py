import cv2
import numpy as np
from ultralytics.utils.plotting import colors

from collections import defaultdict
import pandas as pd
import math


def read_csv_and_calculate_mean(csv_path: str, chunk_size: int = 24):
    df = pd.read_csv(csv_path)
    num_points = len(df)
    num_chunks = num_points // chunk_size
    mean_points = []

    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]
        mean_left_x = chunk['Lx'].mean()
        mean_left_y = chunk['Ly'].mean()
        mean_right_x = chunk['Rx'].mean()
        mean_right_y = chunk['Ry'].mean()

        mean_points.append({
            "left": (mean_left_x, mean_left_y),
            "right": (mean_right_x, mean_right_y)
        })

    return mean_points


def predict(video_path: str, csv1_path: str, csv2_path: str, output_path: str, ids_map: dict):
    csv1_set_of_points = read_csv_and_calculate_mean(csv1_path)
    csv2_set_of_points = read_csv_and_calculate_mean(csv2_path)

    left_track_history = defaultdict(lambda: [])
    right_track_history = defaultdict(lambda: [])

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
                left_point = csv_points['left']
                right_point = csv_points['right']

                if math.isnan(left_point[0]) or math.isnan(left_point[1]):
                    left_point = None
                else:
                    left_point = (int(left_point[0]), int(left_point[1]))

                if math.isnan(right_point[0]) or math.isnan(right_point[1]):
                    right_point = None
                else:
                    right_point = (int(right_point[0]), int(right_point[1]))

                # Store tracking history
                left_track = left_track_history[track_id]
                right_track = right_track_history[track_id]

                if left_point is not None:
                    left_track.append(left_point)
                if right_point is not None:
                    right_track.append(right_point)

                if len(left_track) > 3:
                    left_track.pop(0)
                if len(right_track) > 3:
                    right_track.pop(0)

                # Plot tracks (left)
                text_add = False
                if len(left_track) > 0:
                    left_track_points = np.array(left_track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (left_track[-1]), 7, colors(int(track_id), True), -1)
                    cv2.polylines(frame, [left_track_points], isClosed=False, color=colors(int(track_id), True), thickness=2)

                    # Add Text
                    cv2.putText(frame, ids_map[track_id], (left_track[-1][0] + 10, left_track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(track_id), True), 2)
                    text_add = True

                # Plot tracks (right)
                if len(right_track) > 0:
                    right_track_points = np.array(right_track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (right_track[-1]), 7, colors(int(track_id), True), -1)
                    cv2.polylines(frame, [right_track_points], isClosed=False, color=colors(int(track_id), True), thickness=2)

                    # Add Text
                    if not text_add:
                        cv2.putText(frame, ids_map[track_id], (right_track[-1][0] + 10, right_track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors(int(track_id), True), 2)

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = "./output.mp4"
    csv1_path = "./1017735502_ASD.csv"
    csv2_path = "./1029188263_TD.csv"
    output_path = "./tracking1.avi"
    ids_map = {0: "ASD", 1: "TD"}
    predict(video_path=video_path, csv1_path=csv1_path, csv2_path=csv2_path, output_path=output_path, ids_map=ids_map)


if __name__ == '__main__':
    main()
