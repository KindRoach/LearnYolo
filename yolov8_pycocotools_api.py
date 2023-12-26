from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2ltwh

model = YOLO("yolov8x.pt")

dataDir = "dataset/coco2017"
data_type = "val"
val_anno = COCO(f"{dataDir}/annotations/instances_{data_type}2017.json")
class2id_coco = {cat["name"]: cat["id"] for cat in val_anno.cats.values()}

res = []
for img in tqdm(val_anno.imgs.values()):
    img_path = f"{dataDir}/{data_type}2017/{img['file_name']}"
    dets = model(img_path, verbose=False)
    for box in dets[0].boxes:
        model_class_id = box.cls.int().item()
        cid = class2id_coco[model.names[model_class_id]]
        res.append({
            "image_id": img["id"],
            "category_id": cid,
            "bbox": xywh2ltwh(box.xywh).flatten().int().tolist(),
            "score": box.conf.item()
        })

# Run the evaluation
dt = val_anno.loadRes(res)
cocoEval = COCOeval(val_anno, dt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
