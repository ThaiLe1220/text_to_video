import os
import random
import shutil

def move_random_files(input_dir, output_dir, num_files=20):
    """
    Randomly select and copy files from input_dir to output_dir.

    Parameters:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        num_files (int): Number of random files to select. Default is 20.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files in the input directory (filter for video or image files)
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi", ".mov", ".mkv")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if len(files) < num_files:
        print(f"Warning: Only {len(files)} files available to copy.")

    # Randomly select files
    selected_files = random.sample(files, min(num_files, len(files)))

    # Copy selected files to the output directory
    for file_name in selected_files:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        shutil.copy(src_path, dest_path)

    print(f"Copied {len(selected_files)} files to {output_dir}.")

# Example usage
input_directory = "videos_100"
output_directory = "videos_20"
move_random_files(input_directory, output_directory)

import cv2
import os
from tqdm import tqdm
import math

def get_frames(video_path, num_frames=1):
    """
    Extracts specified number of evenly spaced frames from a video.

    Parameters:
    - video_path (str): Path to the input video file.
    - num_frames (int): Number of frames to extract (max 4).

    Returns:
    - List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        raise IOError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        raise ValueError("Video has no frames.")

    # Cap the number of frames to 4
    num_frames = min(max(num_frames, 1), 4)

    # Calculate frame indices to extract
    frame_indices = []
    for i in range(num_frames):
        # Evenly spaced indices
        index = math.floor((i + 1) * total_frames / (num_frames + 1))
        frame_indices.append(index)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {idx} from {video_path}")
            continue
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise IOError(f"No frames extracted from {video_path}")

    return frames

def stack_frames(frames):
    """
    Stacks frames together along the longer side.

    Parameters:
    - frames (List[numpy.ndarray]): List of frames to stack.

    Returns:
    - Stacked image as a numpy array.
    """
    # Determine stacking direction based on first frame
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    if width >= height:
        # Stack Vertical
        stacking_axis = 0 
    else:
        # Stack Horizontally
        stacking_axis = 1  

    # Resize all frames to have the same height (for horizontal) or width (for vertical) before stacking
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        if stacking_axis == 1:
            # Horizontal stacking: match heights
            scaling_factor = height / h
            new_width = int(w * scaling_factor)
            new_height = height
        else:
            # Vertical stacking: match widths
            scaling_factor = width / w
            new_height = int(h * scaling_factor)
            new_width = width
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_frames.append(resized_frame)

    # Stack frames
    stacked_image = cv2.hconcat(resized_frames) if stacking_axis == 1 else cv2.vconcat(resized_frames)
    return stacked_image

def resize_image(image, target_size=1120):
    """
    Resizes an image so that its longer side is exactly target_size pixels, maintaining aspect ratio.

    Parameters:
    - image (numpy.ndarray): Image to resize.
    - target_size (int): Size to set the longer side.

    Returns:
    - Resized image as a numpy array.
    """
    height, width = image.shape[:2]
    if width >= height:
        scaling_factor = target_size / width
    else:
        scaling_factor = target_size / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def process_video(video_path, output_path, num_frames=1, target_size=1120):
    """
    Processes a single video: extracts frames, stacks them, resizes, and saves as an image.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_path (str): Path where the output image will be saved.
    - num_frames (int): Number of frames to extract and stack (max 4).
    - target_size (int): Size to set the longer side of the final image.
    """
    try:
        frames = get_frames(video_path, num_frames)
        if len(frames) == 1:
            # Single frame processing
            final_image = resize_image(frames[0], target_size)
        else:
            # Multiple frames stacking
            stacked_image = stack_frames(frames)
            final_image = resize_image(stacked_image, target_size)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the final image
        success = cv2.imwrite(output_path, final_image)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def process_videos(input_dir='videos_10', output_dir='images_10', num_frames=1, target_size=1120):
    """
    Processes all supported videos in the input directory by extracting frames,
    stacking them if needed, resizing, and saving as images in the output directory.

    Parameters:
    - input_dir (str): Directory containing input video files.
    - output_dir (str): Directory where output images will be saved.
    - num_frames (int): Number of frames to extract and stack (max 4).
    - target_size (int): Size to set the longer side of the final image.
    """
    # Ensure the output directory exists before processing
    os.makedirs(output_dir, exist_ok=True)

    # Supported video formats
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg')

    # Filter only supported video files
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not videos:
        print(f"No supported video files found in directory: {input_dir}")
        return

    for video in tqdm(videos, desc="Processing videos"):
        input_video_path = os.path.join(input_dir, video)
        output_image_name = os.path.splitext(video)[0] + '.jpg'
        output_image_path = os.path.join(output_dir, output_image_name)

        process_video(
            video_path=input_video_path,
            output_path=output_image_path,
            num_frames=num_frames,
            target_size=target_size
        )

if __name__ == "__main__":
    # Example usage
    # Set num_frames to the desired number (1 to 4)
    NUM_FRAMES_TO_EXTRACT = 1  # Change this value as needed (1, 2, 3, or 4)
    process_videos(input_dir='videos_20', output_dir='images_20', num_frames=NUM_FRAMES_TO_EXTRACT, target_size=1120)
