import cv2
import mmcv
import numpy as np
from PIL import Image
from mmcv.runner import load_checkpoint
from mmseg.apis import inference
import time
import copy
import torch.nn.functional as F
from torchvision import transforms
import warnings
from custom.models import DTP
import mmcv
import torch
from mmseg.models import build_segmentor


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


def eval_psnr(model, original_image):
    start = time.time()

    resized_image = original_image.resize((512, 512))


    prediction = inference.inference_segmentor(model, np.array(resized_image))

    pred_image = Image.fromarray(prediction[0].astype(np.uint8))


    pred_image = pred_image.resize(original_image.size, Image.NEAREST)


    binary_image = pred_image.point(lambda x: 255 if x > 0.5 else 0)


    img1 = binary_image


    mask_white = binary_image.convert('1')
    img2 = Image.composite(original_image, Image.new('RGB', original_image.size, (0, 0, 0)), mask_white)


    mask_black = binary_image.point(lambda x: 0 if x > 0 else 255).convert('1')
    img3 = Image.composite(original_image, Image.new('RGB', original_image.size, (255, 0, 0)), mask_black)


    end = time.time()
    print(end - start)

    return img1, img2, img3


def run(img):
    model_path = 'day.pth'
    cfg_path = 'water_cfg.py'
    cfg = mmcv.Config.fromfile(cfg_path)
    # checkpoint = load_checkpoint(model, model_path, map_location='cpu')
    model = inference.init_segmentor(cfg, model_path)

    return eval_psnr(model, img)


if __name__ == '__main__':
    import gradio as gr

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# 水体分割")
        # gr.Markdown("### Open-World Detection with Grounding DINO")

        with gr.Row():
            with gr.Column():
                img_path = gr.Image(source='upload', type="pil")
                run_button = gr.Button(label="Run")

            with gr.Column():
                iimg = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
                img = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
                mask = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=False, full_height=False)
        run_button.click(fn=run, inputs=[
            img_path], outputs=[mask, img, iimg])

    gr.interface
    block.launch(server_name='0.0.0.0', server_port=7579)
