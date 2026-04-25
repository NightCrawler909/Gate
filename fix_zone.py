import cv2
import json
import os

points = []
frame_copy = None

def click_event(event, x, y, flags, params):
    global frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append({"x": float(x), "y": float(y)})
        # Draw a dot
        cv2.circle(frame_copy, (x, y), 4, (0, 255, 0), -1)
        # Connect the dots
        if len(points) > 1:
            prev = points[-2]
            cv2.line(frame_copy, (int(prev['x']), int(prev['y'])), (x, y), (0, 255, 0), 2)
        cv2.imshow('Map Correct Zone', frame_copy)

# 1. Load the ACTUAL video frame
cap = cv2.VideoCapture('sample.avi')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read sample.avi. Make sure the file is in this directory.")
    exit()

# Make a copy to draw on safely
frame_copy = frame.copy()

print("\n--- INSTRUCTIONS ---")
print("1. Click the corners of the red restricted zone on the ground.")
print("2. Press 'c' to close the polygon.")
print("3. Press 's' to save and exit.")
print("--------------------\n")

cv2.imshow('Map Correct Zone', frame_copy)
cv2.setMouseCallback('Map Correct Zone', click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'c' to close
    if key == ord('c') and len(points) > 2:
        first = points[0]
        last = points[-1]
        cv2.line(frame_copy, (int(last['x']), int(last['y'])), (int(first['x']), int(first['y'])), (0, 255, 0), 2)
        cv2.imshow('Map Correct Zone', frame_copy)
        print("Polygon closed! Press 's' to save.")
        
    # Press 's' to save
    elif key == ord('s'):
        os.makedirs('data', exist_ok=True)
        config = {"restricted_zone": points}
        with open('data/zone_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("SUCCESS: New zone_config.json saved perfectly!")
        break
        
    # Press 'q' to quit
    elif key == ord('q'):
        break

cv2.destroyAllWindows()