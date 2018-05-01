# Mohammad Saad
# utils.py
# Various utilities

def center_crop(image, cropX, cropY):
    y, x, d = image.shape

    startX = x // 2 - (cropX // 2)
    startY = y // 2 - (cropY // 2)

    return image[startY:startY+cropY, startX:startX+cropX, :]
