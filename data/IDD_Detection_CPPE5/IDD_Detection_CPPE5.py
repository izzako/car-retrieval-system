import os
import json
import datasets

_CITATION = """\
Your dataset citation, if any.
"""

_DESCRIPTION = """\
Custom IDD Detection dataset in COCO-like format, where each annotation contains objects with id, category, bbox, and area.
"""

_HOMEPAGE = "https://huggingface.co/datasets/your-username/your-dataset-name"

_LICENSE = "MIT"

class IDDDetection(datasets.GeneratorBasedBuilder):
    """Custom dataset for IDD Detection in COCO-like format."""
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        id2label_path = os.path.join(self.config.data_dir, "id2label.json")
        with open(id2label_path, "r") as f:
            id2label = json.load(f)
        label_names = [id2label[str(i)] for i in range(len(id2label))]
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "image_id": datasets.Value("int64"),
                "filename": datasets.Value("string"),
                "image": datasets.Image(),
                "height": datasets.Value("int32"),
                "width": datasets.Value("int32"),
                "objects": datasets.Sequence({
                    "id": datasets.Value("int64"),
                    "category": datasets.Value("int32"),
                    "bbox": datasets.Sequence(datasets.Value("float32"),length=4),
                    "area": datasets.Value("float32"),
                }),
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.manual_dir  # this is the path passed to `load_dataset()`
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images_path": os.path.join(data_dir,"images", "train"),
                    "annotations_path": os.path.join(data_dir,"annotations", "train.json"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images_path": os.path.join(data_dir,"images", "val"),
                    "annotations_path": os.path.join(data_dir,"annotations", "val.json"),
                },
            ),
        ]

    def _generate_examples(self, annotations_path, images_path):
        
        with open(annotations_path, "r") as f:
            data = json.load(f)

        for idx, ann in enumerate(data):
            # image_id = ann["image_id"]
            filename = ann["filename"]
            image_filename = f"{filename}.jpg"  # adjust if using .png or other

            image_path = os.path.join(images_path, image_filename)

            objects = []
            for i in range(len(ann["objects"]["id"])):
                obj = {
                    "id": ann["objects"]["id"][i],
                    "category": ann["objects"]["category"][i],
                    "bbox": [float(x) for x in ann["objects"]["bbox"][i]],
                    "area": float(ann["objects"]["area"][i]),
                }
                objects.append(obj)

            yield idx, {
                "image": image_path,
                "image_id": ann["image_id"],
                "filename":filename,
                "width": ann["width"],
                "height": ann["height"],
                "objects": objects,
            }