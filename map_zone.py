import cv2
import json
import os
import sys

# Initialize list to hold coordinates
points = []

def click_event(event, x, y, flags, params):
    global frame
    # Left click records the point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append({"x": x, "y": y})
        # Draw a dot and connect lines as you click
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        if len(points) > 1:
            prev = points[-2]
            cv2.line(frame, (prev['x'], prev['y']), (x, y), (0, 0, 255), 2)
        cv2.imshow('Map Restricted Zone', frame)

# 1. Try to load the video frame
cap = cv2.VideoCapture('sample.mp4')
ret, temp_frame = cap.read()
cap.release()

# 2. Safely copy the memory buffer or fallback to the screenshot
if ret:
    # .copy() prevents OpenCV from deleting the memory when cap is released
    frame = temp_frame.copy() 
else:
    print("Failed to load sample.mp4. Falling back to the uploaded screenshot...")
    # Use the screenshot you uploaded earlier as a fallback
    frame = cv2.imread('screenshot.png') 
    
    if frame is None:
        print("Error: Neither 'sample.mp4' nor 'screenshot.png' was found.")
        sys.exit()

print("Instructions:")
print("1. Click the corners of the red restricted zone.")
print("2. Press 'c' to close the polygon.")
print("3. Press 's' to save and exit.")

cv2.namedWindow('Map Restricted Zone', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Map Restricted Zone', 1280, 720)
cv2.imshow('Map Restricted Zone', frame)
cv2.setMouseCallback('Map Restricted Zone', click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'c' to draw the final closing line of the polygon
    if key == ord('c') and len(points) > 2:
        first = points[0]
        last = points[-1]
        cv2.line(frame, (last['x'], last['y']), (first['x'], first['y']), (0, 0, 255), 2)
        cv2.imshow('Map Restricted Zone', frame)
        print("Polygon closed.")
        
    # Press 's' to save the coordinates to JSON
    elif key == ord('s'):
        os.makedirs('data', exist_ok=True)
        config = {"restricted_zone": points}
        with open('data/zone_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Successfully saved to data/zone_config.json")
        break
        
    # Press 'q' to quit without saving
    elif key == ord('q'):
        break

cv2.destroyAllWindows()