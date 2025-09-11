import cv2
import numpy as np

def segment_contours(img_np):
    
    #Segmentation using contour detection.
    #Works well for single & multiple digits, but may struggle with connected ones.
    
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        digit = thresh[y:y+h, x:x+w]
        digit_resized = cv2.resize(digit, (28, 28))
        digits.append(digit_resized)
    return digits


def segment_connected(img_np):
    
    #Segmentation using connected components.
    #Good for single, multiple, and sometimes connected digits.
    
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    digits = []
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        digit = thresh[y:y+h, x:x+w]
        digit_resized = cv2.resize(digit, (28, 28))
        digits.append(digit_resized)
    return digits


def segment_projection(img_np):
    # Segmentation using vertical projection profile.
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(thresh, axis=0)  # vertical projection
    cuts = []
    in_digit = False
    start = 0

    for i, val in enumerate(projection):
        if val > 0 and not in_digit:
            in_digit = True
            start = i
        elif val == 0 and in_digit:
            in_digit = False
            cuts.append((start, i))

    # handle digit that goes till the right edge
    if in_digit:
        cuts.append((start, len(projection)-1))

    digits = []
    for (s, e) in cuts:
        # ignore noise or tiny cuts
        if e - s > 5:
            digit = thresh[:, s:e]
            digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(digit_resized)

    return digits

