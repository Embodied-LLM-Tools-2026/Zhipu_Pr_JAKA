import numpy as np
import cv2
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

class YOLOEProcessor:
    def __init__(self, model_path="weights/yoloe-11l-seg.pt", text_prompt=None):
        """
        Initialize YOLOE batch processor
        Args:
            model_path: YOLOE model path
        """
        self.model = YOLOE(model_path)
        if text_prompt:
            self.set_text_prompt(text_prompt)
        self.warm_up(num_warmup=3)
        
    def set_text_prompt(self, text_prompt):
        """Set text prompt"""
        self.text_prompt = text_prompt
        names = [text_prompt]
        self.model.set_classes(names, self.model.get_text_pe(names))
        
    def warm_up(self, num_warmup=10):
        """Model warm-up using a 640x480 zero array"""
        print(f"Warming up the model, running inference {num_warmup} times...")
        # Create a 640x480 zero array for warm-up
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(num_warmup):
            self.model.predict(dummy_image)
        print("Warm-up completed!")
        
    def process_images(self, img, visual_prompts=None, refer_image=None, save_path=None):
        """
        Batch process image detection and segmentation
        
        Args:
            img: Input image, can be str | np.ndarray | PIL.Image.Image
        """

        # Run detection
        if visual_prompts:
            results = self.model.predict(img, visual_prompts=visual_prompts, refer_image=refer_image, predictor=YOLOEVPSegPredictor)
        else:
            results = self.model.predict(img)
        if len(results) == 0:
            return None
        else:
            #print(f"Detected {len(results)} results, only processing the first one")
            result = results[0]
        
        if save_path:
            result.save(save_path)

        original_height, original_width = result.orig_shape
        masks = self._extract_masks(result, original_width, original_height)

        return masks
        
        
    def _extract_masks(self, result, original_width, original_height):
        """Extract mask information"""
        
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if hasattr(masks, 'data') and masks.data is not None:
                mask_data = masks.data.cpu().numpy()

                # 合并所有检测到的mask
                combined_mask = np.zeros_like(mask_data[0], dtype=np.uint8)
                
                for i in range(mask_data.shape[0]):
                    # 获取当前mask
                    current_mask = mask_data[i]
                    
                    # 转换为布尔数组
                    mask_bool = current_mask > 0.5
                    
                    # 将当前mask添加到合并mask中
                    combined_mask = np.logical_or(combined_mask, mask_bool).astype(np.uint8)
                
                # 调整mask到原始图像大小
                mask_resized = cv2.resize(
                    combined_mask.astype(np.float32), 
                    (original_width, original_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # 重新阈值化调整后的mask
                mask_final = (mask_resized > 0.5).astype(np.uint8)
                
                return mask_final
            
            else:
                return None
        else:
            return None


def main():
    # Set input, can be path, ndarray, or PIL.Image.Image
    img = r"D:\VisualDetector\datasets\drink\drinks\维他们_拉伸4.jpg"
    # img = r"D:\VisualDetector\datasets\drink\drink_ref\维他_侧视.jpg"
    text_prompt = "the brown cuboid"  
    # Model path
    model_path=r"D:\VisualDetector\weights\yoloe-11l-seg.pt"
    # box prompt
    visual_prompts = dict(
        # bboxes=np.array([[164, 107, 297, 476]]),  # 可乐
        # bboxes=np.array([[148, 95, 289, 470]]),  # 雪碧
        # bboxes=np.array([[129, 33, 274, 531]]),  # 奶茶
        # bboxes=np.array([[142, 152, 351, 471]]),  # 维他
        # bboxes=np.array([[135, 96, 364, 365]]),  # 维他_侧视
        bboxes=np.array([[300, 212, 345, 282]]),  # 维他们
        cls=np.array([0]),  # ID to be assigned for person
    )
    refer_image = r"D:\VisualDetector\datasets\drink\drink_ref\维他们.jpg"

    # Create processor and perform batch processing
    processor = YOLOEProcessor(model_path, text_prompt=None)
    masks = processor.process_images(img, visual_prompts=visual_prompts, refer_image=refer_image, save_path=r"D:\VisualDetector\tmp\result.png")
    # print(masks.shape)

if __name__ == "__main__":
    main()