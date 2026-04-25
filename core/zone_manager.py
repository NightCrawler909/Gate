import cv2
import numpy as np
import json

class ZoneManager:
    def __init__(self, config_path='data/zone_config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if list of dicts or list of lists
        zone_data = config.get('restricted_zone', [])
        if zone_data and isinstance(zone_data[0], dict):
            pts = [[pt['x'], pt['y']] for pt in zone_data]
        else:
            pts = zone_data
            
        self.polygon = np.array(pts, np.int32)
        self.polygon = self.polygon.reshape((-1, 1, 2))

    def is_inside(self, bbox_bottom_center):
        x, y = bbox_bottom_center
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), measureDist=False)
        return result >= 0

    def distance_to_polygon(self, bbox_bottom_center):
        x, y = bbox_bottom_center
        # measureDist=True returns distance to closest contour edge
        dist = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), measureDist=True)
        return abs(dist)

    def draw_zone(self, frame):
        cv2.polylines(frame, [self.polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        return frame
