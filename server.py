from flask import Flask, render_template
import torch
from open_generative_fill import config
from open_generative_fill.lm_models import run_lm_model
from open_generative_fill.load_data import load_image
from open_generative_fill.vision_models import (
    run_caption_model,
    run_inpainting_pipeline,
    run_segmentaiton_pipeline,
)
import matplotlib.pyplot as plt
from flask import Flask, render_template


def processGenImageAI(url, prompt):
    image_url = url
    edit_prompt = prompt
    seed_value = 41

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GENERATOR = torch.Generator().manual_seed(seed_value)
    image = load_image(image_url=image_url, image_size=config.IMAGE_SIZE)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()

    # caption = run_caption_model(
    #     model_id=config.CAPTION_MODEL_ID, image=image, device=DEVICE
    # )

    # print("#" * 100)
    # print(f"Using caption model: {config.CAPTION_MODEL_ID}")
    # print(f"Caption: {caption}")
    # print("#" * 100)

    # print(f"{edit_prompt=}")

    # to_replace, replaced_caption = run_lm_model(
    #     model_id=config.LANGUAGE_MODEL_ID,
    #     caption=caption,
    #     edit_prompt=edit_prompt,
    #     device=DEVICE,
    # )

    # print("#" * 100)
    # print(f"Using llm: {config.LANGUAGE_MODEL_ID}")
    # print(f"Object to replace: {to_replace}")
    # print(f"Caption for inpaint: {replaced_caption}")
    # print("#" * 100)

    # segmentation_mask = run_segmentaiton_pipeline(
    #     detection_model_id=config.DETECTION_MODEL_ID,
    #     segmentation_model_id=config.SEGMENTATION_MODEL_ID,
    #     to_replace=to_replace,
    #     image=image,
    #     device=DEVICE,
    # )

    # print("#" * 100)
    # print(f"Using detection model: {config.DETECTION_MODEL_ID}")
    # print(f"Using segmentation model: {config.SEGMENTATION_MODEL_ID}")
    # print("#" * 100)

    # output = run_inpainting_pipeline(
    #     inpainting_model_id=config.INPAINTING_MODEL_ID,
    #     image=image,
    #     mask=segmentation_mask,
    #     replaced_caption=replaced_caption,
    #     image_size=config.IMAGE_SIZE,
    #     generator=GENERATOR,
    #     device=DEVICE,
    # )

    # print("#" * 100)
    # print(f"Using diffusion model: {config.INPAINTING_MODEL_ID}")
    # print("#" * 100)
    print(str(image))

    return "<img src='" + url + "' alt='image ai'/>"

    # plt.figure(figsize=(5, 5))
    # plt.imshow(output)
    # plt.axis("off")
    # plt.show()


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/ai")
def ai():
    return processGenImageAI(
        "https://www.calliaweb.co.uk/wp-content/uploads/2015/10/600x600.jpg",
        "Change 600 is 700",
    )


app.run()
