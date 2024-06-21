import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

import os
import pandas as pd


def predict(video_path: str, export_filepath: str):
    track_history = defaultdict(lambda: [])
    model = YOLO("yolov8n.pt")
    names = model.model.names

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(
        f"{video_path}_annotations.avi",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    tracking_results = list()
    frame_number = -1
    while cap.isOpened():
        success, frame = cap.read()
        frame_number += 1

        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
                    annotator.box_label(box,
                                        color=colors(int(cls), True),
                                        label=f"n: {names[int(cls)]} - c: {conf:.2f} - id: {track_id}")

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                    # Aggregate tracking results
                    tracking_results.append({
                        "frame_number": frame_number,
                        "track_id": int(track_id),
                        "label": names[int(cls)],
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "width": int(box[2] - box[0]),
                        "height": int(box[3] - box[1]),
                        "confidence": conf
                    })

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # export the results
    tracking_results.sort(key=lambda x: x["frame_number"])
    df = pd.DataFrame(tracking_results)
    df.to_csv(export_filepath, index=False)

    result.release()
    cap.release()
    cv2.destroyAllWindows()


# TODO: Find model for the eye tracker (maybe 1D Conv)
def main():
    video_path = os.path.join(os.getcwd(), "pexels-thirdman-5538262 (1080p).mp4")
    export_filepath = f"{video_path}_annotations.csv"

    predict(video_path=video_path, export_filepath=export_filepath)


if __name__ == '__main__':
    main()
