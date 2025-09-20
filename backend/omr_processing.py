import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

def pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        images.append(img)
    return images

def preprocess_omr(img):
    """Correct rotation, skew, perspective."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(sheet_contour, True)
    approx = cv2.approxPolyDP(sheet_contour, 0.02 * peri, True)
    if len(approx) == 4:
        rect = order_points(np.array([pt[0] for pt in approx], dtype="float32"))
        dst = np.array([[0,0],[800,0],[800,1000],[0,1000]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(img, M, (800,1000))
        return warp
    return img

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_bubbles(img, rows=20, cols=5):
    """Detect marked/unmarked bubbles using pixel fill ratio."""
    h, w = img.shape[:2]
    bubble_h, bubble_w = h//rows, w//cols
    responses = []

    for i in range(rows):
        for j in range(cols):
            x, y = j*bubble_w, i*bubble_h
            roi = img[y:y+bubble_h, x:x+bubble_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            fill_ratio = cv2.countNonZero(binary)/binary.size
            responses.append(1 if fill_ratio > 0.6 else 0)
    return responses

def score_omr(responses, answer_key):
    subject_scores = []
    for s in range(5):
        start, end = s*20, (s+1)*20
        score = sum([1 if responses[i]==answer_key[i] else 0 for i in range(start,end)])
        subject_scores.append(score)
    total_score = sum(subject_scores)
    return {"subject_scores": subject_scores, "total_score": total_score}

def annotate_sheet(img, responses, rows=20, cols=5):
    h, w = img.shape[:2]
    bubble_h, bubble_w = h//rows, w//cols
    annotated = img.copy()
    for i in range(rows):
        for j in range(cols):
            x, y = j*bubble_w, i*bubble_h
            if responses[i*cols+j]:
                cv2.rectangle(annotated, (x,y), (x+bubble_w, y+bubble_h), (0,255,0), 2)
    return annotated
