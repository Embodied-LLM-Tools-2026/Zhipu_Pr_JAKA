import os
import cv2
import numpy as np
from .SAM_Seg import SAMProcessor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

class SAMAnnotator:
    def __init__(self, model_type="sam"):
        """
        Initialize SAM annotator
        
        Args:
            model_type: Model type ("mobile_sam", "sam", "fastsam")
        """
        self.processor = SAMProcessor(model_type=model_type)
        self.image = None
        self.original_image = None
        self.mask = None
        self.segmentation = None
        self.clicked_points = []
        self.fig = None
        self.ax = None
        self.current_image_path = None
    
    def mask_to_coco(self, mask: np.ndarray):
        mask_area = np.sum(mask)
        
        if mask_area < 64 * 64:
            kernel_size = (3, 3)  # use smaller kernel for small mask
        else:
            kernel_size = (7, 7)  # use larger kernel for large mask

        blurred = cv2.GaussianBlur(mask * 255, kernel_size, 0)
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        segmentation = []
        simplified_contours = []
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        for contour in contours:
            # hierarchical simplification (smaller, simpler)
            area = cv2.contourArea(contour)
            r = 0.01 if area < 32 * 32 else 0.005 if area < 64 * 64 else 0.003 if area < 96 * 96 else 0.002 if area < 128 * 128 else 0.001
            epsilon = r * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # adjust epsilon if less than 3 points
            while contour.size >= 6 and len(approx) < 3:
                r *= 0.5  # decrease ratio
                epsilon = r * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(approx)
            approx_area = cv2.contourArea(approx)
            if approx.size >= 6 and approx_area / mask_area >= 0.05:
                # 归一化处理
                norm_points = []
                for pt in approx:
                    x, y = pt[0]
                    norm_points.extend([float(x) / w, float(y) / h])
                segmentation.append(norm_points)

        if len(segmentation) == 0:
            # delete empty mask annotations after simplification
            return {}
        else:
            return segmentation
        
    def load_image(self, image_path):
        """
        Load image
        
        Args:
            image_path: Image file path
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert to RGB format
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image = self.original_image.copy()
        
        # Save current image path
        self.current_image_path = image_path
        
        print(f"Successfully loaded image: {image_path}")
        print(f"Image size: {self.image.shape}")
        
    def on_click(self, event):
        """
        Mouse click event handler
        
        Args:
            event: matplotlib click event
        """
        if event.inaxes != self.ax:
            return
        
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Check if coordinates are within image bounds
        if self.image is not None and 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            print(f"Click coordinates: ({x}, {y})")
            
            # Add click point
            self.clicked_points.append([x, y])
            
            # Process image with SAM
            self.process_click(x, y)
            
    def process_click(self, x, y):
        """
        Process click event, generate mask
        
        Args:
            x: x coordinate
            y: y coordinate
        """
        try:
            print("Generating segmentation mask...")
            
            # Use SAM processor to generate mask
            mask, box = self.processor.process_image(
                self.image, 
                points=[[x, y]], 
                labels=[1],  # 1 indicates foreground point
                get_box=True
            )
            
            if mask is not None:
                self.mask = mask
                print("Mask generated successfully!")
                
                # Convert to COCO format
                print("Converting to COCO format...")
                self.segmentation = self.mask_to_coco(mask)
                if self.segmentation:
                    print(f"Conversion successful, obtained {len(self.segmentation)} segmentation regions")
                else:
                    print("Segmentation is empty after conversion")

                self.box = box
                
                # Perform visualization
                self.visualize_result()
                
                # Save results
                self.save_segmentation_to_txt()
                
            else:
                print("Cannot generate mask, please try another location")
                
        except Exception as e:
            print(f"Error processing click: {str(e)}")
    
    def visualize_result(self):
        """
        Visualize segmentation results (display COCO converted contours and box if available)
        """
        if self.ax is None or self.original_image is None:
            print("Image not loaded")
            return
        
        # Clear previous image
        self.ax.clear()
        
        # Display original image
        self.ax.imshow(self.original_image)
        
        # If segmentation data exists, draw COCO converted contours
        if self.segmentation is not None and len(self.segmentation) > 0:
            # Get image dimensions for denormalization
            h, w = self.original_image.shape[:2]
            
            for i, seg in enumerate(self.segmentation):
                # Convert segmentation data to coordinate points and denormalize
                coords = np.array(seg).reshape(-1, 2)
                # Denormalize: x * w, y * h
                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                
                # Draw contour
                self.ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2, 
                           label=f'Contour {i+1}' if i == 0 else "")
                
                # Fill contour area (semi-transparent)
                from matplotlib.patches import Polygon
                polygon = Polygon(coords, facecolor='red', alpha=0.3, edgecolor='red')
                self.ax.add_patch(polygon)
        
        # Visualize box if available
        if hasattr(self, 'box') and self.box is not None:
            if (isinstance(self.box, (list, tuple)) and len(self.box) == 4) or (
                isinstance(self.box, np.ndarray) and self.box.shape == (4,)):
                x1, y1, x2, y2 = self.box
                width = x2 - x1
                height = y2 - y1
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='blue', facecolor='none', label='Box')
                self.ax.add_patch(rect)
                # Show box values at the top-left corner of the box
                box_text = f"Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                self.ax.text(x1, y1 - 10, box_text, color='blue', fontsize=10, backgroundcolor='white')
        
        # Display click points
        for i, point in enumerate(self.clicked_points):
            self.ax.plot(point[0], point[1], 'go', markersize=8, label=f'Click Point {i+1}')
        
        # Set title and labels
        self.ax.set_title('SAM Annotation (COCO Contours)')
        self.ax.legend()
        self.ax.axis('off')
        
        # Refresh display
        if self.fig is not None:
            self.fig.canvas.draw()
    
    def save_segmentation_to_txt(self):
        """
        Save current segmentation data to txt file
        """
        if self.segmentation is None:
            print("No segmentation data to save")
            return
        
        try:
            # Get current image filename (without extension)
            if self.current_image_path is not None:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                # Create txt file path with same name as image
                txt_path = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}.txt")
                
                # Write segmentation to txt file
                with open(txt_path, 'w') as f:
                    for i, seg in enumerate(self.segmentation):
                        # Convert segmentation data to string format
                        seg_str = ' '.join([str(x) for x in seg])
                        f.write(f"0 {seg_str}\n")
                
                print(f"Segmentation saved to: {txt_path}")
                print(f"Contains {len(self.segmentation)} segmentation regions")
                
            else:
                print("Current image path not found, cannot save segmentation")
                
        except Exception as e:
            print(f"Error saving segmentation: {str(e)}")
        
    def reset_annotation(self):
        """
        Reset annotation
        """
        self.mask = None
        self.segmentation = None
        self.clicked_points = []
        
        if self.ax is not None and self.original_image is not None:
            self.ax.clear()
            self.ax.imshow(self.original_image)
            self.ax.set_title('SAM Segmentation - Click image to annotate, press q to quit, press r to reset')
            self.ax.axis('off')
            
            if self.fig is not None:
                self.fig.canvas.draw()
        
        print("Annotation reset")
    
    def save_result(self, output_path):
        """
        Save annotation results
        
        Args:
            output_path: Output file path
        """
        if self.mask is None:
            print("No mask to save")
            return
        
        # Save mask as numpy array
        np.save(output_path.replace('.png', '_mask.npy'), self.mask)
        
        # Save visualization results
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
    
    def run_annotation(self, image_path, output_path=None):
        """
        Run interactive annotation
        
        Args:
            image_path: Input image path
            output_path: Output result path (optional)
        """
        # Load image
        self.load_image(image_path)
        
        # Create matplotlib figure
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display original image
        if self.original_image is not None:
            self.ax.imshow(self.original_image)
            self.ax.set_title('SAM Segmentation - Click image to annotate, press q to quit, press r to reset')
            self.ax.axis('off')
        
        # Bind events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Keyboard event handling
        def on_key(event):
            if event.key == 'q':
                plt.close()
                return
            elif event.key == 'r':
                self.reset_annotation()
            elif event.key == 's' and output_path:
                self.save_result(output_path)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("=" * 50)
        print("SAM Interactive Annotation Tool")
        print("=" * 50)
        print("Instructions:")
        print("- Click anywhere on the image to perform segmentation")
        print("- Press 'r' key to reset annotation")
        print("- Press 's' key to save results (if output path is specified)")
        print("- Press 'q' key to quit program")
        print("=" * 50)
        
        # Display figure
        plt.show(block=True)

def main():
    """
    Main function - Batch annotate all images in folder
    """
    import glob
    
    # Configuration parameters
    model_type = "sam"  # Options: "mobile_sam", "sam", "fastsam"
    
    # Set input and output paths
    input_folder = r"D:\VisualDetector\datasets\robot_view"
    output_folder = r"D:\VisualDetector\datasets\robot_view_anno"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"No image files found in folder {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Create annotator
    annotator = SAMAnnotator(model_type=model_type)
    
    # Process each image sequentially
    for i, image_path in enumerate(image_files):
        annotator.reset_annotation()
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # Set output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_result.png")
            
            # Run annotation
            annotator.run_annotation(image_path, output_path)
            
            print(f"Image {os.path.basename(image_path)} annotation completed")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
    
    print(f"\nBatch annotation completed! Processed {len(image_files)} images")
    print(f"Results saved in: {output_folder}")
    print(f"Segmentation data saved in original image folder")

if __name__ == "__main__":
    main() 