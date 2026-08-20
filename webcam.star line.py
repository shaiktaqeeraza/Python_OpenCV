import cv2
 
# ---- Global state for the mouse-driven drawing ----
drawing = False           # True while the left mouse button is held down
start_point = (-1, -1)    # Point where the current drag started
current_line = None       # The line being dragged right now (preview)
lines = []                # All finalized lines: list of (start, end) tuples
 
 
def draw_line(event, x, y, flags, param):
    """Mouse callback: tracks click-drag-release to build a line."""
    global drawing, start_point, current_line, lines
 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_line = None
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_line = (start_point, (x, y))
 
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        lines.append((start_point, (x, y)))
        current_line = None
 
 
def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
 
    if not cap.isOpened():
        print("Error: Could not access the webcam. Check that it's connected "
              "and not being used by another application.")
        return
 
    window_name = "Webcam - Draw Lines"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_line)
 
    print("Instructions:")
    print(" - Click and drag with the left mouse button to draw a line")
    print(" - Press 'c' to clear all lines")
    print(" - Press 'q' to quit")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break
 
        # Mirror the frame so it feels natural (like a mirror)
        frame = cv2.flip(frame, 1)
 
        # Draw all lines that have already been finalized
        for (start, end) in lines:
            cv2.line(frame, start, end, (0, 255, 0), 2)
 
        # Draw the line currently being dragged (live preview), in red
        if current_line is not None:
            cv2.line(frame, current_line[0], current_line[1], (0, 0, 255), 2)
 
        cv2.imshow(window_name, frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            lines.clear()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 