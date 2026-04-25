from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        # class 0 = person. class 24 = backpack, 26 = handbag
        results = self.model.predict(frame, classes=[0, 24, 26], verbose=False)
        return results[0]

    def track(self, frame, persist=True):
        results = self.model.track(frame, classes=[0, 24, 26], persist=persist, verbose=False)
        return results[0]
