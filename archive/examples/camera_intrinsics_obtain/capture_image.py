import cv2
import os

def capture_images(save_dir='./images', num_images=5, camera_id=0):
    """
    拍摄指定数量的摄像头图像并保存到指定文件夹
    Args:
        save_dir (str): 保存图片的文件夹
        num_images (int): 拍摄图片数量
        camera_id (int): 摄像头编号，默认0
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按空格拍照，按q退出。")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # 空格键拍照
            img_path = os.path.join(save_dir, f'image_{count+1}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"已保存: {img_path}")
            count += 1
        elif key == ord('q'):
            print("用户退出。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("拍摄结束。")

if __name__ == "__main__":
    capture_images()
