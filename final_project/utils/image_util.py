from PIL import Image

def convert_to_list(IMAGE_PATH, size: int = 64) -> list:
    img = Image.open(IMAGE_PATH).convert('RGB')
    img = img.resize(size=(size, size))
    width, height = img.size
    pixels = list(img.getdata())
    rows = []
    for i in range(height):
        columns = []
        for j in range(width):
            columns.append(list(pixels[i * width + j]))
        rows.append(columns)
    pixel_matrix = rows
    return pixel_matrix

def shape(pixel_matrix):
    return (len(pixel_matrix), len(pixel_matrix[0]), len(pixel_matrix[0][1]))