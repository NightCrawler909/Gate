from collections import deque
import numpy as np

class RiskScorer:
    def __init__(self, history_size=10):
        # Rolling average queue dict
        self.active_tracks = {}
        self.history_size = history_size

    def calculate_risk(self, person_id, features_dict):
        # Init new id tracking
        if person_id not in self.active_tracks:
            self.active_tracks[person_id] = deque(maxlen=self.history_size)
            
        # Parse features
        speed_var = features_dict.get('speed_variance', 0.0)
        time_near = features_dict.get('time_near_boundary', 0.0)
        stops = features_dict.get('sudden_stops', 0)
        baggage = features_dict.get('carrying_baggage', 0)
        dist = features_dict.get('distance_to_restricted_zone', 100.0)
        time_mod = features_dict.get('time_of_day_multiplier', 1.0)
        
        # Raw weights sum
        raw_score = (speed_var * 2.0) + (time_near * 1.5) + (stops * 3.0) + (baggage * 10.0)
        
        # Zone penalty
        if dist <= 0:
            raw_score += 85.0
            
        # Global multiplier & cap limit
        final_score = min(100.0, raw_score * time_mod)
        
        self.active_tracks[person_id].append(final_score)
        return float(np.mean(self.active_tracks[person_id]))
        
    def remove_track(self, person_id):
        # Cleanup exit frame ids
        if person_id in self.active_tracks:
            del self.active_tracks[person_id]
