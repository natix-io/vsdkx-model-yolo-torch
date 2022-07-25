## YOLOv5

This project supports models that are trained on the [YOLOv5](https://github.com/ultralytics/yolov5) architecture implemented on `pytorch`.

By default, we load the model's custom weights in `YoloTorchDriver()` (`vsdk/model/yolo_torh/driver.py`) using the hub of `torch`, as shown below:

```python
self._yolo = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=model_config['model_path'])
```

Furthermore, `YoloTorchDriver()` supports inferencing with method `inference(self, frame_object: FrameObject) -> Inference`. This method performs the following three steps:
1. Preprocesses the input frame by resizing it to the required input size by yolo
2. Inferences the resized frame 
3. Processes the inference results, performs NMS and filters bounding boxes by their class IDs (if any ids are gives in `filter_class_ids` in the `model_config`)


### Model Settings

The below model settings can be customized and passed to the `YoloTorchDriver()` as a `model_settings` dictionary from the outside world.

```yaml
conf_thresh: 0.5, # Float class confidence threshold
iou_thresh: 0.4, # Float Intersection of Union threshold
device: 'cpu', # Device string used for pytorch (options: 'cpu'| 'gpu')
```

Please mind that if no `model_settings` are passed to `YoloTorchDriver()`, it will fall into the default settings stored in `vsdk/model/yolo_torch/settings.yaml`.

### Model Config

The below model config settings are **required** for the `YoloTorchDriver()` to run. They are passed as a `model_config` dictionary and should not be customizable from the outside world.

```yaml
input_shape: # Input image shape
filter_class_ids: # Class IDs that should be kept upon inference
classes_len: # Amount of classes the model predicts
model_path: # Path to the model
```