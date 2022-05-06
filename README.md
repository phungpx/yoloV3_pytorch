# YoloV3

![yolov3](https://user-images.githubusercontent.com/86842861/143871002-b2516c01-b3d2-4d2a-b1fe-62bfb28bcb47.gif)

## 1. Project Structure
```
yoloV3_pytorch
      |
      ├── config
      |	    ├── electric_labelme_training.yaml      
      |	    └── electric_labelme_testing.yaml
      |
      ├── flame
      |	    ├── core 
      |     |     ├── data 
      |     |     |     ├── electric_components.py
      |     |     |     └── visualize.py
      |     |     |
      |     |     ├── engine
      |     |     |     ├── evaluator.py
      |     |     |     └── trainer.py
      |     |     |
      |     |     ├── loss
      |     |     |     ├── loss.py
      |     |     |     └── yolov3_loss.py
      |     |     |
      |     |     └── model
      |     |           ├── darknet53.py
      |     |           └── model.py
      |     |     
      |     |     
      |	    └── handle               
      |           ├── metrics
      |           |     ├── loss
      |           |     |     ├── loss.py
      |           |     |     └── yolov3_loss.py
      |           |     |
      |           |     ├── mean_average_precision
      |           |     |     ├── evaluator.py
      |           |     |     └── mean_average_precision.py
      |           |     |
      |           |     └── metrics.py
      |           |
      |           ├── checkpoint.py
      |           ├── early_stopping.py
      |           ├── lr_scheduler.py
      |           ├── metric_evaluator.py
      |           ├── region_predictor.py
      |           ├── screenlogger.py
      |           └── terminate_on_nan.py
      |   
      |
      ├── __main__.py
      ├── module.py
      └── utils.py
```
## 2. Dataset
* Data about 200 images labeled (json file) of "Electric Components".
* Data divided into 2 parts: train and valid folders.

| Name  | Train | Valid | Test | Label's Format |
| ---   | ---         |     ---      |  --- |   --- |
| Electric Components | 152 |  50    |  ---   | JSON    |

## 3. Model & Metrics
![image](https://user-images.githubusercontent.com/61035926/167065201-7f02bce6-1e15-44e3-8545-8b0d576446f3.png)
- I used YOLOv3 and also use pretrain model trained with "Electric Components" dataset to greatly reduce training time.
- In training process, I used "Learning Rate Schedule" and "Early Stopping" to adjust learning rate follow loss value (3 epochs) and stop training when loss unimprove passing some epochs (10 epochs).

## 4. How to Run
### Clone github
* Run the script below to clone my github.
```
git clone https://github.com/PhongPX1603/detect_electric_components.git
```

### Training
* Trained by pytorch-ignite framework. Install: ```pip install pytorch-ignite```
* Dataset structure
```
dataset
    ├── train
    │   ├── img1.jpg
    │   ├── img1.json
    |   ├── img2.jpg
    |   ├── img2.json
    │   └── ...
    │   
    └── valid
        ├── img1.jpg
        ├── img1.json
        ├── img2.jpg
        ├── img2.json
        └── ...
```
* Change your direct of dataset folder in ```config/electric_labelme_training.yaml```
* Run the script below to train the model. Specify particular name to identify your experiment:
```python -m flame configs/electric_labelme_training.yaml```

### Evaluation
* Change your direct of test dataset folder in ```config/electric_labelme_testing.yaml```
* Run the script below to train the model. Specify particular name to identify your experiment:
```python -m flame configs/electric_labelme_testing.yaml```

## 4. Inference
* Download Weight
```bash
https://drive.google.com/drive/folders/1mmJsO71xahQ9gVjPdKUpQAVm1_SXOeQp?usp=sharing
```
* You can use this script to make inferences on particular folder
* Result are saved at <output/img.jpg> if type inference is 'image' or <video-output.mp4> with 'video or webcam' type.
```bash
cd inference
python real_time_inference.py --type-inference 'image' --input-dir <image dir> --video-output <video_output.mp4>
                                               'video'             <video dir>
                                               'webcam'            0
```
* You can use this script to make inferences on particular folder
* Result are saved at <output/img.jpg> if type inference is 'image' or <video-output.mp4> with 'video or webcam' type.
```
cd inference
python real_time_inference.py --type-inference 'image' --input-dir <image dir> --video-output <video_output.mp4>
                                               'video'             <video dir>
                                               'webcam'            0
```

## 5. Feature
* YOLOv4

## 6. Acknowledgements
* https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/model.py

## Contributor
*Xuan-Phung Pham*
