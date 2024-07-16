import dtlpy as dl
import json


def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    loaded_item = dl.Item.from_json(_json=data, client_api=dl.client_api)
    if isinstance(loaded_item.annotations, list) and len(loaded_item.annotations) > 0:
        annotations = loaded_item.annotations
    else:
        annotations = loaded_item.annotations_link

    for annotation in annotations:
        annotation = dl.Annotation.from_json(_json=annotation, client_api=dl.client_api)
        print(
            f"Annotation ID: {annotation.id}, "
            f"Label: {annotation.label}, "
            f"Type: {annotation.type}, "
            f"Coordinates: {annotation.coordinates}"
        )


def main():
    json_path = "./girls_new.json"
    load_annotations(json_path=json_path)


if __name__ == '__main__':
    main()
