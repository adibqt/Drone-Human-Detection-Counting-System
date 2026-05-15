# Trained weights

The Colab-trained YOLOv8s checkpoint lives here as `best.pt`.

After running `notebooks/task02_train_colab.ipynb`, copy
`MyDrive/drone-detection/runs/yolov8s_visdrone/weights/best.pt` into
this folder. Everything in the local pipeline (`src/eval`, `src/infer`)
defaults to `outputs/weights/best.pt`.
