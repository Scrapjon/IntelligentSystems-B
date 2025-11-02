import cv2
import numpy as np

def segment_contours(img_np):
    """
    Segmentation using contour detection.
    This version scales the digit to fit a 20x20 box and centers it 
    in a 28x28 canvas to match MNIST formatting.
    """
    
    # Invert the image: MNIST is white digit (255) on black background (0)
    # Your canvas is black digit (0) on white background (255)
    # THRESH_BINARY_INV handles this inversion.
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore small noise contours
        if w < 5 or h < 5:
            continue

        # Crop the digit from the thresholded image
        digit_crop = thresh[y:y+h, x:x+w]

        # --- MNIST-style Preprocessing ---
        
        # 1. Create a new 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)

        # 2. Resize the cropped digit to fit within a 20x20 box, maintaining aspect ratio
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Ensure dimensions are at least 1x1
        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1

        digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. Calculate coordinates to center the 20x20 box in the 28x28 canvas
        # (4px padding on each side)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2

        # 4. Paste the resized digit onto the center of the canvas
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = digit_resized
        
        digits.append(canvas)
        
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

