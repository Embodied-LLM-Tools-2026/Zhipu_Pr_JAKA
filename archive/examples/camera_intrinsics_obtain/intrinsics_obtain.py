import cv2
import numpy as np
import glob
import tyro
import json


def main(
            square_size: int
):

    # 设置棋盘格尺寸（内角点数量）
    checkerboard_size = (11, 8)  # 行和列的内角点数量

    # 定义停止条件
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备对象点，例如 (0,0,0), (24,0,0), (48,0,0), ...，单位是 square_size
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                        0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D 点
    imgpoints = []  # 2D 点

    # 获取所有标定图像的文件路径
    images = glob.glob(r'./images/*.jpg')  # 根据实际情况修改路径和文件类型
    print(images)
    for fname in images:
        # 读取图像
        img = cv2.imread(fname)

        print('--------------------------------')
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            # 提高角点精度
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 可视化角点
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"角点未找到：{fname}")

    cv2.destroyAllWindows()

    # 获取图像尺寸
    img_shape = gray.shape[::-1]

    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    # 打印和保存结果
    print("标定是否成功：", ret)
    print("相机内参矩阵：\n", camera_matrix.tolist())
    print("畸变系数：\n", dist_coeffs)

    camera_intrinsics = {'camera': camera_matrix.tolist()}
    with open('camera_new.json', 'w') as file:
        json.dump(camera_intrinsics, file, indent=4)


    # 计算总的误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error

    print("总的平均误差：", total_error/len(objpoints))

if __name__ == "__main__":
    tyro.cli(main)
