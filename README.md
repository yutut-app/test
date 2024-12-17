# 明るい領域と暗い領域の周辺の変化を強調
gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)  # CV_64FからCV_32Fに変更
gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)  # CV_64FからCV_32Fに変更

# メモリ効率の良い計算方法に変更
gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)  # np.sqrtの代わりにcv2.magnitudeを使用
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
