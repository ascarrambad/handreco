
import cv2
import numpy as np

# Count number of files to increment new example directory
_gesture_list = []
with open('gestures.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line != '':
            _gesture_list.append(line)

def draw_inferences(values):
    nb_classes              = len(values)
    left_margin             = 150
    margin                  = 50
    thickness               = 40

    font                    = cv2.FONT_HERSHEY_SIMPLEX
    fontScale               = 1
    fontColor               = (255,255,255)
    lineType                = 2

    blank = np.zeros((350,600,3), np.uint8)

    for i in range(nb_classes):
        if(values[i] > 0.7):
            cv2.rectangle(blank, (left_margin, margin + int(margin*i)), (left_margin + int(values[i]*200), margin + thickness + int(margin*i)), (0,255,0), -1)
        else:
            cv2.rectangle(blank, (left_margin, margin + int(margin*i)), (left_margin + int(values[i]*200), margin + thickness + int(margin*i)), (255,0,0), -1)
        cv2.putText(blank, _gesture_list[i], (0, margin + int(margin*i) + int(thickness/2)), font, fontScale, fontColor, lineType)
        cv2.putText(blank, str(values[i]), (left_margin + 200, margin + int(margin*i) + int(thickness/2)), font, fontScale, fontColor, lineType)

    cv2.imshow("Inferences", blank)
    cv2.waitKey(1)