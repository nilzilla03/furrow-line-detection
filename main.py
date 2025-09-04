import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import os


def DFT(hist):
    """Compute first two DFT coefficients of the histogram."""
    F1, F2 = 0, 0
    N = len(hist)
    W_N = np.exp(-2j * np.pi / N)

    for i in range(N):
        F1 += np.power(W_N, i) * hist[i]
        F2 += np.power(W_N, 2 * i) * hist[i]

    return np.real(F1), np.real(F2)


def compute_max_diff_img(img, window=25):
    """Compute binary image of max difference positions row-wise."""
    Y, X = img.shape
    img_blur = cv2.GaussianBlur(img, (5, 5), 0).astype(np.int64)
    max_diff_img = np.zeros((Y, X), dtype=np.uint8)

    for i in range(Y):
        row_max = 0
        max_pos = 0
        for j in range(window, X - window):
            sum_r = np.sum(img_blur[i, j + 1 : j + window + 1])
            sum_l = np.sum(img_blur[i, j - window : j])
            diff = sum_l - sum_r

            if diff > row_max:
                row_max = diff
                max_pos = j

        max_diff_img[i, max_pos] = 255

    return max_diff_img


def find_lines(img, main_img, thresh, theta_low, theta_high):
    """Detect and draw only the first valid line using Hough transform."""
    lines = cv2.HoughLines(img, 1, np.pi / 180, thresh)
    if lines is None:
        print("No lines detected")
        return main_img, None, None

    for r_theta in lines:
        r, theta = r_theta[0]
        if theta_low < theta < theta_high:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * r, b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(main_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return main_img, r, theta

    return main_img, None, None


def main(image_path, output_dir, window, hough_thresh, theta_low, theta_high):
    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load grayscale + color image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)

    if img_gray is None or img_color is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save grayscale
    gray_out = os.path.join(output_dir, f"{base_name}_gray.png")
    cv2.imwrite(gray_out, img_gray)

    # Histogram & DFT
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist != 0]
    F1, F2 = DFT(hist)
    print("F2 > F1:", F2 > F1)

    # Max diff image
    max_diff_img = compute_max_diff_img(img_gray, window=window)
    max_diff_out = os.path.join(output_dir, f"{base_name}_maxdiff.png")
    cv2.imwrite(max_diff_out, max_diff_img)

    # Line detection
    lines_detected, r, theta = find_lines(max_diff_img, img_color.copy(), hough_thresh, theta_low, theta_high)
    lines_out = os.path.join(output_dir, f"{base_name}_lines.png")
    cv2.imwrite(lines_out, lines_detected)

    if r is not None:
        print(f"Line detected: r={r:.2f}, theta={theta:.2f} rad")
    else:
        print("No valid line found.")

    cv2.imshow("Grayscale", img_gray)
    cv2.imshow("Max Diff Image", max_diff_img)
    cv2.imshow("Lines Detected", lines_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect lines from image using histogram + DFT + Hough.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder to save output images (default: outputs)")
    parser.add_argument("--window", type=int, default=25, help="Window size for difference computation (default: 25)")
    parser.add_argument("--hough_thresh", type=int, default=10, help="HoughLines threshold (default: 10)")
    parser.add_argument("--theta_low", type=float, default=2.7, help="Lower bound for theta in radians (default: 2.7)")
    parser.add_argument("--theta_high", type=float, default=2.95, help="Upper bound for theta in radians (default: 2.95)")

    args = parser.parse_args()
    main(args.image_path, args.output_dir, args.window, args.hough_thresh, args.theta_low, args.theta_high)
