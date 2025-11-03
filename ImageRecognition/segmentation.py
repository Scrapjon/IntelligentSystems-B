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
        scale = 28.0 / max_dim
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

# In segmentation.py
import cv2
import numpy as np
# ... (your other segmentation functions) ...


def segment_watershed(img_np, erosion = 2, dilation = 3):
    """
    Segmentation using the watershed algorithm to separate touching digits.
    """
    # 1. Binarize the image (white digit on black background)
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)

    # --- This is the critical tuning section ---
    
    # 2. Find "sure foreground" (cores of the digits) by eroding
    # A smaller kernel (3x3) and fewer iterations are sensitive.
    # More iterations will separate stubborn digits but might erase thin ones (like "1").
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(thresh, kernel, iterations=2)

    # 3. Find "sure background" by dilating
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    
    # --- End tuning section ---

    # 4. Find the "unknown" region (the boundaries where basins will meet)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Create markers for the watershed algorithm
    # Labels background as 0, and other objects as 1, 2, 3...
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that the "sure background" is 1, not 0
    markers = markers + 1

    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # 6. Run the watershed algorithm
    # It requires a 3-channel (color) image as input
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # Boundaries are now marked with -1 in 'markers'
    
    # Invert the original grayscale image to match MNIST (white-on-black)
    # We will crop from this to preserve grayscale data for the SVC model
    img_inverted_grayscale = 255 - img_np

    digits = []
    
    # Loop through all found labels (1 is background, -1 is boundary)
    for label in np.unique(markers):
        if label <= 1:
            continue
        
        # Create a mask for the current label
        mask = np.zeros(img_np.shape, dtype="uint8")
        mask[markers == label] = 255

        # Find the contour of this segmented object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Get the bounding box of the segmented digit
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Ignore noise
        if w < 5 or h < 5:
            continue

        # --- Crop from the INVERTED GRAYSCALE image ---
        digit_crop = img_inverted_grayscale[y:y+h, x:x+w]
        
        # --- MNIST-style Preprocessing (Center and Pad) ---
        canvas = np.zeros((28, 28), dtype=np.uint8)
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1

        digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2

        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = digit_resized
        
        digits.append(canvas)

    return digits

