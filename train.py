import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/yolov8.yaml')
    # model.load('yolov8n.pt')lytics/cfg/ # loading pretrain weights
    model.train(
                data='ultralytics/cfg/datasets/Bump.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=1,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )