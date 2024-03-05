import cv2
import numpy as np
import torch
import torchvision
from easyocr import easyocr
from torchvision.transforms import functional as F
from estimate_watermark import estimate_watermark, crop_watermark, poisson_reconstruct, PlotImage
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from read_data import read_video_to_frames


def read_video():
    path = "../watermark/1.mp4"

    frames = torchvision.io.read_video(path)[0]

    same_indexes = torch.ones_like(frames[0]).bool()
    for i, frame in enumerate(frames[1:]):
        same_indexes = same_indexes * (frame == frames[i-1])

    test_frame = same_indexes * frames[0]

    F.to_pil_image(test_frame.permute(2,0,1)).save("test.png")


def gradient_and_crop(frames):
    gx, gy, gxlist, gylist = estimate_watermark(frames)
    xm, xM, ym, yM = crop_watermark(gx, gy)
    cropped_gx, cropped_gy = gx[xm:xM, ym:yM, :], gy[xm:xM, ym:yM, :]
    W_m = poisson_reconstruct(cropped_gx, cropped_gy)

    W_m_image = (PlotImage(W_m)*255).astype(np.uint8)
    return W_m_image

    #im, start, end = watermark_detector(frames[10], cropped_gx, cropped_gy)

    #cropped_frames = get_cropped_frames(frames, start, end)


def recognize_text(image):
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def watermark_recognition(input_file="3.mp4"):
    input_folder = "../watermark/"
    order = input_file.split(".")[0]
    frames = read_video_to_frames(input_folder + input_file)
    watermark = gradient_and_crop(frames)
    watermark_name = f"watermark_{order}.png"
    cv2.imwrite(watermark_name, watermark)

    reader = easyocr.Reader(['en'])  # Specify the languages you want to recognize

    # Detect and recognize text in the image
    results = reader.readtext(watermark_name)

    # Print the recognized text
    if len(results) != 0:
        print(f"watermark with text detected: {results[0][1]}")
    else:
        print("no watermark detected")


if __name__ == "__main__":
    watermark_recognition("8.mov")
