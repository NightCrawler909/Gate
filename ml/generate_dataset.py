import pandas as pd
import numpy as np

def generate_dataset(num_rows=5000, output_path='custom_dataset.csv'):
    np.random.seed(42)
    data = {
        'speed_variance': np.random.uniform(0.0, 5.0, num_rows),
        'distance_to_restricted_zone': np.random.uniform(0.0, 500.0, num_rows),
        'time_near_boundary': np.random.randint(0, 121, num_rows),
        'time_inside_zone': np.random.randint(0, 61, num_rows),
        'sudden_stops': np.random.randint(0, 11, num_rows),
        'carrying_baggage': np.random.choice([0, 1], num_rows),
        'time_of_day_multiplier': np.random.uniform(1.0, 2.5, num_rows)
    }
    df = pd.DataFrame(data)
    intents, scopes = [], []
    for _, row in df.iterrows():
        if row['time_inside_zone'] > 0:
            intent = 'active_intrusion'
        elif row['time_near_boundary'] > 20 and row['sudden_stops'] > 3:
            intent = 'scouting'
        elif row['time_near_boundary'] > 15:
            intent = 'loitering'
        else:
            intent = 'passing_by'
        intents.append(intent)
        
        raw_risk = (
            row['speed_variance'] * 2.0 +
            row['time_near_boundary'] * 1.5 +
            row['sudden_stops'] * 3.0 +
            row['carrying_baggage'] * 10.0
        )
        if row['distance_to_restricted_zone'] <= 0:
            raw_risk += 85.0
        
        final_risk = min(100.0, raw_risk * row['time_of_day_multiplier'])
        scopes.append(round(final_risk, 2))
        
    df['intent_class'] = intents
    df['historical_risk_score'] = scopes
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    generate_dataset()
