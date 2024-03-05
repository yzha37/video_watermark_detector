import pytesseract
from PIL import Image
import easyocr


if __name__ == "__main__":
    # Open the image
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the languages you want to recognize

    # Load the image
    image = 'watermark.png'

    # Detect and recognize text in the image
    results = reader.readtext(image)

    # Print the recognized text
    for result in results:
        text = result[1]
        print(text)