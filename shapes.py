import cv2
import numpy as np

# Create a blank canvas
img = np.ones((600, 800, 3), dtype=np.uint8) * 255

drawing = False
shape = "rectangle"
ix, iy = -1, -1


# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, img

    # Mouse button pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Mouse button released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        if shape == "rectangle":
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 2)

        elif shape == "circle":
            radius = int(((x - ix)**2 + (y - iy)**2)**0.5)
            cv2.circle(img, (ix, iy), radius, (0, 0, 255), 2)


# Create window
cv2.namedWindow("Draw Shapes")
cv2.setMouseCallback("Draw Shapes", draw)

print("Press:")
print("R - Rectangle")
print("C - Circle")
print("E - Erase Screen")
print("Q - Quit")

while True:
    cv2.imshow("Draw Shapes", img)

    key = cv2.waitKey(1) & 0xFF

    # Select Rectangle
    if key == ord('r'):
        shape = "rectangle"
        print("Rectangle selected")

    # Select Circle
    elif key == ord('c'):
        shape = "circle"
        print("Circle selected")

    # Clear the canvas
    elif key == ord('e'):
        img[:] = 255

    # Quit
    elif key == ord('q'):
        break

cv2.destroyAllWindows()