import gdown
import os
import cv2
from glob import glob
from dotenv import load_dotenv
load_dotenv()

test_video = "test_video.mp4"
FRAME_DIR = "frame_dir"
PRED_FRAME_DIR = "pred_frame_dir"
OUTPUT_NAME = "output.mp4"
OUTPUT_VID_NAME = "output_video.mp4"

def video_to_images(video_path, output_folder, prefix="frame", image_format="jpg"):
    """
    Extracts frames from a video file and saves them as images.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Define the output image path
        image_name = f"{prefix}_{frame_count:05d}.{image_format}"
        image_path = os.path.join(output_folder, image_name)

        # Save the frame as image
        cv2.imwrite(image_path, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


def images_to_video(image_folder, output_path, video_length_seconds=197, fps=None):
    """
    Converts a series of images in a folder into a video.

    Args:
        image_folder (str): Path to the folder containing image files.
        output_path (str): Path to save the output video file (e.g., 'output.mp4').
        video_length_seconds (int or float): Desired video length in seconds.
        fps (int, optional): If specified, overrides automatic FPS calculation.
    """
    # Get and sort image file paths
    image_files = sorted(glob(os.path.join(image_folder, "*.*")))
    num_images = len(image_files)

    if num_images == 0:
        raise ValueError("No images found in the provided folder.")

    # Load the first image to get the size
    sample_img = cv2.imread(image_files[0])
    height, width, _ = sample_img.shape

    # Determine FPS (frames per second)
    if fps is None:
        fps = num_images / video_length_seconds
    else:
        video_length_seconds = num_images / fps

    print(f"Creating video of {video_length_seconds:.2f} seconds at {fps:.2f} FPS.")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change to 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame.shape[0:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    #downlaod the video file using gdown
    
    vid_id = os.environ.get("VIDEO_URL")
    gdown.download(id=vid_id, output=test_video, quiet=False)


    # convert the video to images
    video_to_images(test_video, FRAME_DIR)

    images_to_video(PRED_FRAME_DIR, OUTPUT_VID_NAME)