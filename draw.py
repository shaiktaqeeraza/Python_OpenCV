import cv2
import numpy as np

# White canvas
img = np.ones((700, 600, 3), dtype=np.uint8) * 255

step = 0

def draw_mario(event, x, y, flags, param):
    global step

    if event == cv2.EVENT_LBUTTONDOWN:

        if step == 0:
            # Face
            cv2.circle(img, (300, 180), 60, (180, 220, 255), -1)

        elif step == 1:
            # Hat
            cv2.rectangle(img, (235, 100), (365, 145), (0, 0, 255), -1)
            cv2.rectangle(img, (215, 140), (385, 155), (0, 0, 255), -1)

        elif step == 2:
            # Eyes
            cv2.circle(img, (280, 175), 6, (0,0,0), -1)
            cv2.circle(img, (320, 175), 6, (0,0,0), -1)

        elif step == 3:
            # Nose
            cv2.ellipse(img, (300,200), (12,18), 0, 0, 360, (170,200,255), -1)

        elif step == 4:
            # Mustache
            cv2.ellipse(img, (300,220), (30,10), 0, 0, 180, (0,0,0), -1)

        elif step == 5:
            # Mouth
            cv2.ellipse(img, (300,245), (15,8), 0, 0, 180, (0,0,255), 2)

        elif step == 6:
            # Shirt
            cv2.rectangle(img, (255,240), (345,340), (255,0,0), -1)

        elif step == 7:
            # Arms
            cv2.line(img, (255,260), (210,310), (180,220,255), 12)
            cv2.line(img, (345,260), (390,310), (180,220,255), 12)

        elif step == 8:
            # Overalls
            cv2.rectangle(img, (260,290), (340,390), (255,0,0), -1)
            cv2.line(img, (260,290), (280,240), (255,0,0), 8)
            cv2.line(img, (340,290), (320,240), (255,0,0), 8)

        elif step == 9:
            # Legs
            cv2.line(img, (280,390), (280,470), (150,75,0), 12)
            cv2.line(img, (320,390), (320,470), (150,75,0), 12)

        elif step == 10:
            # Shoes
            cv2.ellipse(img, (270,490), (20,10), 0, 0, 360, (0,0,0), -1)
            cv2.ellipse(img, (330,490), (20,10), 0, 0, 360, (0,0,0), -1)

        step += 1

cv2.namedWindow("Draw Mario")
cv2.setMouseCallback("Draw Mario", draw_mario)

while True:
    cv2.imshow("Draw Mario", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()