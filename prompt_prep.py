import os
import re
import sys
import time
from openai import OpenAI


def process_input_gpt_thai(client, text, model="gpt-4o-mini"):
    system_prompt = (
        "You are an assistant designed to process user input and generate concise and accurate descriptions for a video or image. "
        "Your responsibilities include:\n"
        "1. Remove any NSFW or disallowed content (e.g., violence, hate speech) to maintain a safe and appropriate description.\n"
        "2. Condense lengthy inputs while retaining all key descriptive elements essential for clarity.\n"
        "3. If known figures or cultural references appear, include their names and provide a brief description of their physical appearance or distinguishing features.\n"
        "4. Output only in English, avoiding any additional commentary or explanations.\n"
        "### User Input:\n"
    )

    user_prompt = f"{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=100,
        timeout=15,
    )

    processed_text = re.sub(
        r"\s*\n\s*", " ", response.choices[0].message.content.strip()
    )
    return processed_text


def process_input_gpt_huy(client, text, model="gpt-4o-mini"):
    system_prompt = (
        "You are an assistant designed to process user input and generate concise and accurate descriptions for video or image. "
        "Your responsibilities include:\n"
        "1. Remove any NSFW or disallowed content to maintain a safe and appropriate description.\n"
        "2. Condense lengthy inputs while retaining all key descriptive elements essential for clarity.\n"
        "3. Output only in English, avoiding any additional commentary or explanations.\n"
        "### User Input:\n"
    )

    user_prompt = f"{text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=120,
            timeout=15,
        )

        processed_text = re.sub(
            r"\s*\n\s*", " ", response.choices[0].message.content.strip()
        )
        return processed_text

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def process_input_gpt(client, text, model="gpt-4o-mini"):

    user_prompt = (
        "Translate the following text into English, return the translated content only, and do not add anything else: "
        + text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract and clean the translated text
        processed_text = response.choices[0].message.content.strip()

        return processed_text

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def main(input_file_path, output_file_path):
    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        sys.exit(1)

    # Initialize the OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Open the input and output files
    try:
        with open(input_file_path, "r", encoding="utf-8") as infile, open(
            output_file_path, "w", encoding="utf-8"
        ) as outfile:

            for idx, line in enumerate(infile, start=1):
                original_text = line.strip()
                if not original_text:
                    print(f"Line {idx} is empty. Skipping.")
                    outfile.write("\n")  # Preserve line numbering
                    continue

                print(f"Translating line {idx}: {original_text}")

                translated_text = process_input_2(client, original_text)

                if translated_text is not None:
                    outfile.write(translated_text + "\n")
                    print(f"Translated text written to line {idx}: {translated_text}'")
                else:
                    outfile.write("\n")  # Write empty line for failed translations
                    print(
                        f"Failed to translate line {idx}. Written empty line to maintain order."
                    )

                # Optional: Sleep to respect API rate limits
                time.sleep(1)  # Adjust sleep time as needed based on your rate limits
    except IOError as e:
        print(f"An I/O error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python translate_lines.py <input_file.txt> <output_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    main(input_file, output_file)
