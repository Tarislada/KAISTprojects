import os
import shutil
import tempfile
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import glob
import tempfile

class VideoProcessor:
    def __init__(self, model_path, video_path, output_video_name=None, image_folder=None, csv_file_path=None, fps=30.0):
        """
        Initialize VideoProcessor with model, paths, and fps.
        
        :param model_path: Path to the YOLO model.
        :param video_path: Path to the input video file.
        :param output_video_name: Name of the output video file.
        :param image_folder: Path to the folder where frame images will be stored.
        :param csv_file_path: Path to save the CSV file.
        :param fps: Frames per second for the output video.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.model = YOLO(model_path)  # Load the YOLO model
        self.video_path = video_path  # Set the input video path
        self.output_video_name = output_video_name  # Set the output video name
        self.generate_video = bool(output_video_name)  # Set self.generate_video based on whether output_video_name is provided or not
        self.image_folder = image_folder if image_folder else tempfile.mkdtemp()  # Set the image folder
        self.csv_file_path = csv_file_path  # Set the path to save the CSV file
        self.fps = fps  # Set the frames per second
        self.results = None  # Initialize results to None
        self.keep_frames = bool(image_folder)  # Set self.keep_frames based on whether image_folder is provided or not
        
        # Future implementation - Determine the image folder path
        # if image_folder:
        #     self.image_folder = Path(image_folder)  # Convert to Path object for more robust path handling
        #     self.keep_frames = True
        # else:
        #     self.image_folder = Path(tempfile.mkdtemp())  # Create a temporary directory and convert to Path object
        #     self.keep_frames = False
        
        # Notify the user of the choices made by the program
        print(f"Image frames will be {'saved' if image_folder else 'discarded after use'}.")
        print(f"Video will be {'generated' if output_video_name else 'not generated'}.")
        print(f"Results will be {'saved to CSV' if csv_file_path else 'not saved to CSV'}.")


    def _process_result(self, result, index):
        """
            This method processes a single result: saves the frame and appends tensors to tensors_list.
            It handles the reshaping and concatenation of tensors to form the desired output structure.
            The reshaping is dynamic, adjusting to the number of keypoints detected.
        """
        # Save frame if required
        if self.generate_video:
            try:
                im = Image.fromarray(result.plot()[...,::-1])
                image_path = os.path.join(self.image_folder, f"{index}.jpg")
                im.save(image_path)
            except IOError as e:  # Catching a more specific error related to I/O operations
                print(f"IOError occurred while saving frame {index}: {e}")
            except Exception as e:  # Catching any other unexpected errors that might occur
                print(f"An unexpected error occurred while saving frame {index}: {e}")

        # Append tensors to tensors_list if required
        if self.csv_file_path:
            try:
                # Key points tensors
                keypoints_xy = result.keypoints.xy
                keypoints_conf = result.keypoints.conf.unsqueeze(-1)  # Add a new axis to make it [1, 11, 1]
                
                # Boxes tensors
                boxes_xywh = result.boxes.xywh
                boxes_conf = result.boxes.conf.unsqueeze(-1)  # Add a new axis to make it [1, 1]
                
                # Create an index tensor
                index_tensor = torch.full((boxes_xywh.shape[0], 1), index, dtype=boxes_xywh.dtype, device=boxes_xywh.device)
                
                # Concatenate index, boxes tensors, keypoints tensors along the last axis
                # Assuming each keypoint has two coordinates (x, y)
                num_keypoints = keypoints_xy.shape[-2]  # Assuming keypoints_xy has a shape [..., num_keypoints, 2]
                concatenated = torch.cat((index_tensor, boxes_xywh, boxes_conf, keypoints_xy.reshape(-1, num_keypoints * 2), keypoints_conf.reshape(-1, num_keypoints)), dim=-1)
                
                # Append the concatenated tensor to the list
                self.tensors_list.append(concatenated)
            except AttributeError as e:
                print(f"AttributeError at result {index}: {e}")        

    def process_video(self):
        """
        Process the input video, save frames, create output video if required, save results to CSV, and clean up.
        """
        if self.keep_frames:
            os.makedirs(self.image_folder, exist_ok=True)

        if self.csv_file_path:
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
        
        try:
            if self.keep_frames and os.path.exists(self.image_folder):
                print(f"Folder {self.image_folder} already exists. Proceeding with overwrite.")
                
            # Perform prediction only once and store the results in an instance variable
            self.results = self.model.predict(self.video_path, stream=True, device=0)
            
            self.tensors_list = []
            for i, result in enumerate(self.results):
                self._process_result(result, i)
                
            if self.tensors_list:
                self._save_csv()
                
            if self.generate_video:
                self._create_video_from_images()
                
            self._clean_up()
            
        except Exception as e:
            print(f"An error occurred while processing the video: {e}")
            
    def _save_csv(self):
        """
        Save the processed tensors to a CSV file.
        """
        try:
            if self.tensors_list:
                final_tensor = torch.cat(self.tensors_list, dim=0)
                np.savetxt(self.csv_file_path, final_tensor.cpu().numpy().reshape(-1, final_tensor.shape[-1]), fmt='%.4f', delimiter=',')
                print(f"Successfully saved results to {self.csv_file_path}")
            else:
                print("No tensors to save to CSV.")
        except Exception as e:
            print(f"An error occurred while saving results to CSV: {e}")
                
    def _create_video_from_images(self):
        """
        Create a video from the saved frame images.
        """
        # Get the list of image files from the image folder
        images = [img for img in os.listdir(self.image_folder) if img.endswith(".png") or img.endswith(".jpg")]
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if not images:
            print("No images found in the specified directory!")
            return

        first_image_path = os.path.join(self.image_folder, images[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"Error reading image {first_image_path}")
            return

        height, width, layers = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_name, fourcc, self.fps, (width, height))

        print(f"Creating video {self.output_video_name}, with {len(images)} frames.")
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.image_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error reading image {image_path}. Skipping this frame.")
                continue

            out.write(image)
            if i % 100 == 0:
                print(f"Processed {i}/{len(images)} frames.")

        out.release()
        print("Video creation is complete.")

    def _clean_up(self):
        """
        Clean up the temporary image folder if keep_frames is False.
        """
        if not self.keep_frames:
            if os.path.exists(self.image_folder):
                try:
                    shutil.rmtree(self.image_folder)
                    print(f"Successfully deleted the temporary image folder: {self.image_folder}")
                except Exception as e:
                    print(f"Error occurred while trying to delete the temporary image folder: {e}")
            else:
                print(f"No folder found at {self.image_folder}. Nothing to delete.")
             
# Usage 
if __name__ == "__main__":
    model_path = '/home/tarislada/YOLOprojects/YOLO_custom/KH_model_v1.pt'
    fps = 30.0
    # # Single file level usage
    
    # video_path = '/home/tarislada/YOLOprojects/YOLO_custom/testvid.mp4'
    # image_folder = '/home/tarislada/YOLOprojects/YOLO_custom/images' # Provide a path if you want to keep the frame images
    # output_video_name = 'M21_20hz_10mw.mp4' # Provide a name if you want to generate a video file
    # csv_file_path = '/home/tarislada/YOLOprojects/YOLO_custom/csv/M21_20hz_10mw_test.csv'  # Provide a path if you want to save results to CSV
    # fps = 30.0
    # # keep_frames = False  # Set to True if you want to keep the frame images
    # # generate_video = False  # Set to False if you don't want to generate a video file
       
    # # processor = VideoProcessor(model_path, video_path, output_video_name, image_folder, csv_file_path, fps)
    # processor = VideoProcessor(model_path=model_path, video_path=video_path, csv_file_path=csv_file_path, fps=fps)
    # processor.process_video()
    
    # Multiple file level usage
    video_directory = '/home/tarislada/YOLOprojects/YOLO_custom/Video_ChR2'
    
    # List all .mp4 files in the specified directory
    video_paths = glob.glob(os.path.join(video_directory, '*.mp4'))

    # Define the model path and other parameters

    # Iterate over all video files and process them
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        # Extract the video file name (without extension) to use it as a base name for output files
        base_name = os.path.basename(video_path)
        name_without_extension = os.path.splitext(base_name)[0]
        
        # Define unique output paths for each video
        output_video_name = f"/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/{name_without_extension}_output.mp4"
        # image_folder = f"/home/tarislada/YOLOprojects/YOLO_custom/images/{name_without_extension}"
        csv_file_path = f"/home/tarislada/YOLOprojects/YOLO_custom/csv/{name_without_extension}.csv"
        
        # Create the processor object and start processing the video
        processor = VideoProcessor(model_path=model_path, video_path=video_path, csv_file_path=csv_file_path, fps=fps)
        processor.process_video()