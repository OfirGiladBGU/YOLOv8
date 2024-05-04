import cv2
from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    print(metrics)


def predict(image_path: str):
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    results = model(image_path)  # predict on an image

    names = model.model.names
    image = cv2.imread(image_path)

    from ultralytics.utils.plotting import Annotator, colors

    boxes = results[0].boxes.xyxy.cpu()

    # Extract prediction results
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.float().cpu().tolist()

    # Annotator Init
    annotator = Annotator(image, line_width=2)

    for box, cls, conf in zip(boxes, clss, confs):
        annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]}-{conf:.2f}")

    # Display annotated image
    annotated_image = annotator.result()
    output_path = "annotated_image.png"
    cv2.imwrite(output_path, annotated_image)

    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_path = "bus.jpg"

    # train()
    predict(image_path=image_path)


if __name__ == '__main__':
    main()
