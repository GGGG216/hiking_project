import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image


# 加载 Midas 模型
def load_midas_model(device):
  
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
 
    model.to(device)
    model.eval()
    return model

# 图像预处理
def preprocess_image(image_path):
    transform = Compose([
        lambda img: img.convert("RGB"),  # 确保图片是 RGB 格式
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path)
    img = img.convert("RGB")  # 如果图片是 RGBA 或其他格式，这里会将其转换为 RGB
    img = img.resize((384, 384))  # 调整到 Midas 的输入尺寸
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # 添加 batch 维度

# 执行深度估计
def estimate_depth(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        depth_map = model(image_tensor)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = depth_map.cpu().numpy()
        return depth_map
def get_depth_map(image_path):
    """
    输入图片路径，返回深度图对应的 NumPy 矩阵。
    
    参数:
        image_path (str): 输入图片的路径。
        
    返回:
        numpy.ndarray: 图片的深度矩阵。
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_midas_model(device)

    # 预处理图片
    image_tensor = preprocess_image(image_path)

    # 生成深度矩阵
    depth_map = estimate_depth(model, image_tensor, device)

    return depth_map
def visualize(depth_map):
    # 可视化深度图
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    print(depth_map_normalized)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow("Depth Map", depth_map_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#主函数
if __name__ == "__main__":
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # 加载模型
    model = load_midas_model(device)
    img_path = "D:\= =\\4544\hiking_project\level4_example_2.jpg"
    image_tensor = preprocess_image(img_path)
    depth_map = estimate_depth(model, image_tensor, device)
    visualize(depth_map)
    # # 输入文件夹路径
    # input_folder = "D:\\= =\\4544\\hiking_project\\final_data\\final_data"  # 替换为包含图片的文件夹路径
    # output_folder = "D:\\= =\\4544\\hiking_project\\data\\depth_maps\\train"  # 替换为保存深度矩阵的文件夹路径

    # # 检查输出文件夹是否存在，不存在则创建
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # # 遍历文件夹中的所有图片
    # for filename in os.listdir(input_folder):
    #     print(filename)
    #     if filename.endswith(".jpg") or filename.endswith(".png"):  # 支持 JPG 和 PNG 格式
    #         input_image_path = os.path.join(input_folder, filename)
    #         output_numpy_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.npy")

    #         print(f"处理图片: {filename}")

    #         # 预处理图片
    #         image_tensor = preprocess_image(input_image_path)

    #         # 生成深度图
    #         depth_map = estimate_depth(model, image_tensor, device)

    #         # 保存为 NumPy 矩阵
    #         np.save(output_numpy_path, depth_map)
    #         print(f"深度图已保存为 NumPy 矩阵: {output_numpy_path}")

    # print("所有图片的深度处理已完成！")

    
