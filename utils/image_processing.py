from PIL import Image, ImageOps


def preprocess_image(image_path, output_size=(600,600)):
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, output_size, Image.LANCZOS)
    img.save(image_path)
    return image_path
