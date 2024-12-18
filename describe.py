# process/describe.py

import ollama
import os
from tqdm import tqdm


def generate_description_from_image(imageURL):
    """
    Generate a detailed description of a video based on 4 stacked key frames.

    Args:
        imageURL (str): The path to the image file containing the stacked key frames.

    Returns:
        str: A structured description of the video content.
    """

    prompt = """
You will be given a single image frame extracted from a short video. Analyze the visual details of this frame and produce a concise English prompt describing what the short video might depict.
Assume the frame is representative of the video, but do not invent details that cannot be seen or logically inferred. If certain information cannot be determined, leave those fields blank.

**Instructions:**
- Output should be in JSON format with the following fields:
  {
    "description": "",
    "main_subject": "",
    "context": "",
    "actions_or_events": "",
    "visual_style": ""
  }

- "description": A concise summary of what the scene depicts.
- "main_subject": The primary subject(s) visible (e.g., a person, animal, object).
- "context": Any environment or setting details you can infer.
- "actions_or_events": Any visible or strongly implied actions.
- "visual_style": Any notable aesthetic aspects (color tone, camera style, etc.) if identifiable.

Provide only the JSON object as the final answer.
"""

    r = ollama.generate(
        model="llama3.2-vision",
        stream=False,
        prompt=prompt,
        options={"seed": 0},
        images=[imageURL],
    )

    return r["response"].replace("\n", " ")


def generate_description_from_images(input_dir, output_dir):
    images = os.listdir(input_dir)
    images.sort()

    metadata_file = f"new_metadata_20.txt"
    last_line = ""
    if not os.path.isfile(metadata_file):
        with open(
            os.path.join(output_dir, metadata_file), "w", encoding="utf-8"
        ) as file:
            file.write("")
    else:
        with open(metadata_file, "r", encoding="utf-8") as file:
            last_line = file.readlines()[-1].split("|")[0]

    file = open(metadata_file, "a", encoding="utf-8")

    unprocess_images = []
    for image in images:
        video = os.path.splitext(image)[0] + ".mp4"
        if video <= last_line:
            continue
        unprocess_images.append(image)

    print(f"Found {len(unprocess_images)} videos without prompts")
    if len(unprocess_images) == 0:
        return

    print(f"\nCreating prompts for videos")
    for image in tqdm(unprocess_images, desc="Describe videos", colour="green"):
        video = os.path.splitext(image)[0] + ".mp4"
        desc = generate_description_from_image(os.path.join(input_dir, image))
        file.write(f"{video}|{desc}\n")
        file.flush()

    file.close()


if __name__ == "__main__":
    generate_description_from_images(input_dir="images_20", output_dir="./")
