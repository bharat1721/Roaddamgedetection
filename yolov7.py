# Copy dataset
cp -r /kaggle/input/uav-china-spain /kaggle/working/dataset

# Clone YOLOv7 repo
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

# Install dependencies
pip install -r requirements.txt



import os

# Fully disable Weights & Biases (wandb)
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'offline'


# Train YOLOv7 model
!python train.py --weights yolov7.pt \
  --cfg cfg/training/yolov7.yaml \
  --data /kaggle/working/dataset/data.yaml \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --device 0 \
  --workers 2 \
  --name yolov7-custom \
  --exist-ok


# Evaluate YOLOv7
!python /kaggle/working/yolov7/test.py \
  --weights /kaggle/working/yolov7/runs/train/yolov7-custom/weights/best.pt \
  --data /kaggle/working/dataset/data.yaml \
  --img-size 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --task val
