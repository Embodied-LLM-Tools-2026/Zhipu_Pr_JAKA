import os
import glob
import time
import numpy as np
import cv2
from ultralytics import SAM, FastSAM

class SAMProcessor:
    def __init__(self, model_type="sam"):
        """
        Initialize SAM processor
        
        Args:
            model_type: Model type ("mobile_sam", "sam", "fastsam")
            model_path: Custom model path, if None use default path
        """
        # print("Initializing SAM processor...")
        
        # Load model
        if model_type == "mobile_sam":
            self.model = SAM("mobile_sam.pt")
        elif model_type == "sam":
            self.model = SAM("sam2.1_b.pt")
        elif model_type == "fastsam":
            self.model = FastSAM("FastSAM-s.pt")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        
        # Show model info
        self.model.info()
        self.warm_up(num_warmup=1)
        # print("SAM processor initialized.")
    
    def warm_up(self, num_warmup=3):
        """Model warm-up using a 640x480 zero array"""
        # print(f"Warming up the model, running inference {num_warmup} times...")
        # Create a 640x480 zero array for warm-up
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(num_warmup):
            self.model(dummy_image)
        print("Warm-up completed!")
    
    def process_image(self, img, points=None, labels=None, save_path=None, get_box=False):
        """
        Process a single image
        
        Args:
            img: Input image, can be str | np.ndarray | PIL.Image.Image
            points: Click prompt points [[x1, y1], [x2, y2], ...]
            labels: Click labels [1, 0, ...] (1 for foreground, 0 for background)
            save_path: Save path, if None do not save
            get_box: Whether to get box
        Returns:
            results: List of segmentation results
        """
        # Run inference
        if points is not None and labels is not None:
            # Use click prompts
            results = self.model(img, points=points, labels=labels)
        else:
            # Default: segment all content
            results = self.model(img)
        
        if len(results) == 0:
            if get_box:
                return None, None
            else:
                return None
        else:
            result = results[0]

        # Save result
        if save_path:
            result.save(save_path)

        
        original_height, original_width = result.orig_shape
        masks = self._extract_masks(result, original_width, original_height)

        if get_box:
            box = self._extract_boxes(result)
            return masks, box
        else:   
            return masks
    
    def process_images_batch(self, input_path, output_path, points=None, labels=None):
        """
        Batch process images
        
        Args:
            input_path: Input image folder path
            output_path: Output result folder path
            points: Click prompt points
            labels: Click labels
            text_prompt: Text prompt
            bboxes: Bounding boxes
        """
        # Create output folder
        os.makedirs(output_path, exist_ok=True)
        
        # Get all PNG and JPG image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        print(f"Found {len(image_files)} image files")
        
        # Warm up
        if len(image_files) > 0:
            self.warm_up(num_warmup=3)
        
        total_time = 0
        
        # Process each image
        for i, image_file in enumerate(image_files):
            try:
                # print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                output_file = os.path.join(output_path, f"{base_name}_result.png")
                
                start_time = time.time()
                
                # Process image
                self.process_image(
                    image_file, 
                    points=points, 
                    labels=labels,
                    save_path=output_file
                )
                
                end_time = time.time()
                
                total_time += end_time - start_time
                print(f"Processing time: {end_time - start_time} seconds")
                    
            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")
                continue
        
        print(f"Batch processing completed! Average time: {total_time / len(image_files)} seconds")
    
    def _extract_masks(self, result, original_width, original_height):
        """Extract mask information"""
        
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if hasattr(masks, 'data') and masks.data is not None:
                mask_data = masks.data.cpu().numpy()

                # Get current mask
                current_mask = mask_data[0]
                
                # Convert mask to boolean array
                mask_bool = current_mask > 0.5
                
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask_bool.astype(np.float32), 
                    (original_width, original_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Re-threshold the resized mask
                mask_final = (mask_resized > 0.5).astype(np.uint8)
                
                return mask_final
            
            else:
                return None
        else:
            return None

    def _extract_boxes(self, result):
        """Extract box information"""
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                box_data = boxes.xyxy.cpu().numpy()

                # Get current box (first box)
                current_box = box_data[0]  # [x1, y1, x2, y2] format
                
                # Extract box coordinates
                x1, y1, x2, y2 = current_box
                
                
                # Return box as [x1, y1, x2, y2] or [x, y, w, h] format
                # You can choose the format based on your needs
                return [x1, y1, x2, y2]  # or return [x, y, w, h]
            
            else:
                return None
        else:
            return None




if __name__ == "__main__":
    # Model config
    model_type = "sam"  # Options: "mobile_sam", "sam", "fastsam"
    
    # Set input and output paths
    img = r"C:\Users\Work\Documents\Pr_Stage1\archive\examples\images\images\image_1.jpg"  # Input image file
    out = r"C:\Users\Work\Documents\Pr_Stage1\archive\examples\images\images\image_100.jpg"   # Output result file

    processor = SAMProcessor(model_type=model_type)
    masks, box = processor.process_image(img, [1381, 1092], [1], save_path=out, get_box=True)
    if masks is not None:
        print(masks.shape)
    else:
        print("No mask generated.")

# Original code kept for reference
# from ultralytics import SAM, FastSAM

# # Load a model
# #model = SAM("sam2.1_b.pt")
# model = SAM("mobile_sam.pt")
# #model = FastSAM("FastSAM-s.pt")

# # Display model information (optional)
# model.info()

# # warm up
# model("/home/higroup/tyz/Project/YOLO-World/demo/sample_images/box_rgb/rbg1.jpg")

# # Run inference
# results = model("/home/higroup/tyz/Project/YOLO-World/demo/sample_images/box_rgb/rbg1.jpg",  points=[1381, 1092], labels=[1])

# # Save results
# results[0].save("/home/higroup/tyz/Project/YOLO-World/tmp/result.png")