## download dataset
download coco 2017 dataset and unzip to under path `dataset`:
- [train2017](http://images.cocodataset.org/zips/train2017.zip)
- [val2017](http://images.cocodataset.org/zips/val2017.zip)
- [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## setup env
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```