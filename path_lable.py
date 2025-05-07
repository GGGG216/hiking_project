import os
import cv2
import numpy as np

IMAGE_FOLDER = "D:\= =\\4544\hiking_project\\new"
BRUSH_RADIUS = 25
ALPHA = 0.5

def get_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith('.jpg') and "_path" not in f]

def paint_mask(image):
    # 缩小图片，避免超大卡顿
    max_side = 1024
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    painting = False

    def mouse_event(event, x, y, flags, param):
        nonlocal painting, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
        elif event == cv2.EVENT_MOUSEMOVE and painting:
            cv2.circle(mask, (x, y), BRUSH_RADIUS, 255, -1)

    cv2.namedWindow('Annotating', cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty('Annotating', cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
    cv2.setMouseCallback('Annotating', mouse_event)

    while True:
        red_layer = np.zeros_like(image)
        red_layer[:, :, 2] = 255
        blended = cv2.addWeighted(image, 1-ALPHA, red_layer, ALPHA, 0)
        mask_3ch = np.stack([mask]*3, axis=2) > 0
        display = image.copy()
        display[mask_3ch] = blended[mask_3ch]
        cv2.imshow('Annotating', display)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            # 返回原尺寸mask
            if scale < 1.0:
                mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                return mask_full
            else:
                return mask
        elif key == ord('n'):
            return None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

def save_path_mask(image, mask, save_path):
    red_layer = np.zeros_like(image)
    red_layer[:, :, 2] = 255
    blended = cv2.addWeighted(image, 1-ALPHA, red_layer, ALPHA, 0)
    mask_3ch = np.stack([mask]*3, axis=2) > 0
    overlay = image.copy()
    overlay[mask_3ch] = blended[mask_3ch]
    cv2.imwrite(save_path, overlay)

def main():
    files = get_image_files(IMAGE_FOLDER)
    for filename in files:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        mask_path = os.path.join(IMAGE_FOLDER, filename.replace('.jpg', '_path.jpg'))
        if os.path.exists(mask_path):
            print(f"{mask_path} exists, skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot open {img_path}, skipping.")
            continue
        print(f"Annotating: {filename}")
        mask = paint_mask(img)
        if mask is not None:
            save_path_mask(img, mask, mask_path)
            print(f"Saved: {mask_path}")
        else:
            print("Skipped.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()