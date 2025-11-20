from collections.abc import Sequence    # 导入Sequence抽象基类，用于类型提示
import logging  # 导入日志模块，用于记录处理过程中的信息

import torch    # 导入PyTorch库，用于张量操作和深度学习功能

from openpi.shared import image_tools  # 导入项目自定义的图像处理工具 

logger = logging.getLogger("openpi")    # 创建日志记录器，用项目名称作为标识

# Constants moved from model.py
# 从model.py移动过来的常量
IMAGE_KEYS = (  # 定义默认的图像键名元组，表示不同相机视角
    "base_0_rgb",   # 基座相机RGB图像
    "left_wrist_0_rgb",  # 左手腕相机RGB图像
    "right_wrist_0_rgb",    # 右手腕相机RGB图像
)

IMAGE_RESOLUTION = (224, 224)   # 定义标准图像分辨率为224x224像素


def preprocess_observation_pytorch(
    observation,    # 输入的观测数据对象
    *,  # 强制使用关键字参数
    train: bool = False,    # 是否处于训练模式，默认为False
    image_keys: Sequence[str] = IMAGE_KEYS, # 要处理的图像键名列表，默认使用IMAGE_KEYS常量
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,   # 目标图像分辨率，默认使用IMAGE_RESOLUTION常量
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    """PyTorch兼容版本的观测数据预处理函数，使用简化的类型注解以避免torch.compile问题。
    该函数对机器人观测数据进行预处理，包括图像标准化和可选的数据增强。
    参数:
        observation: 包含图像和状态数据的观测对象
        train: 布尔值，指示是否处于训练模式，训练模式下会应用数据增强
        image_keys: 要处理的图像键名序列，默认为预定义的IMAGE_KEYS
        image_resolution: 目标图像分辨率，默认为(224, 224)
    返回:
        SimpleProcessedObservation: 包含处理后数据的简化观测对象
    异常:
        ValueError: 当observation.images中缺少指定的image_keys时抛出
    """

    # 检查观测数据中是否包含所有需要的图像键
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 获取批次形状，排除最后一个维度（特征维度）
    batch_shape = observation.state.shape[:-1]

    # 初始化输出图像字典
    out_images = {}
    for key in image_keys:
        image = observation.images[key] # 获取当前图像

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        # 处理不同的图像格式：[B, C, H, W] 和 [B, H, W, C]
        # 通过检查第二个维度是否为3（通道数）来判断格式
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1 检查通道是否在第1维（索引为1）

        # 如果是通道优先格式，转换为通道末尾格式以便处理
        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)   # 将[B, C, H, W]转换为[B, H, W, C]以方便处理

        # 如果图像大小与目标分辨率不匹配，则调整图像大小
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            # 使用带填充的调整大小方法保持纵横比
            image = image_tools.resize_with_pad_torch(image, *image_resolution)
            
        # 如果处于训练模式，应用数据增强
        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            # 将图像从[-1, 1]范围转换到[0, 1]范围，以便进行PyTorch增强
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            # 对非手肘相机应用几何增强
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                # 获取图像高度和宽度
                height, width = image.shape[1:3]

                # Random crop and resize
                # 随机裁剪和调整大小参数设置
                crop_height = int(height * 0.95)    # 裁剪高度为原高度的95%
                crop_width = int(width * 0.95)  # 裁剪宽度为原宽度的95%

                # Random crop
                # 执行随机裁剪
                max_h = height - crop_height     # 计算高度方向上的最大偏移量
                max_w = width - crop_width      # 计算宽度方向上的最大偏移量
                if max_h > 0 and max_w > 0:     # 确保有足够空间进行裁剪
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    # 使用张量操作而非.item()以保持torch.compile兼容性
                    # 随机生成裁剪起始位置
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    # 执行裁剪操作
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                # 将裁剪后的图像调整回原始大小
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w] 转换为pytorch插值函数所需的格式
                    size=(height, width),    # 目标大小为原始尺寸
                    mode="bilinear",         # 使用双线性插值
                    align_corners=False,     # 不对齐角点，避免边缘伪影
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c],转换回原始格式

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                # 随机小角度旋转
                # 使用张量操作而非.item()以保持torch.compile兼容性
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees 随机角度在-5到5度之间
                # 只有当角度足够大时才进行旋转，避免无意义的计算
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians 将角度转换为弧度
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    # 创建旋转矩阵所需的正弦和余弦值
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    # 使用grid_sample应用旋转
                    # 创建网格坐标
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid # 创建网格
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension # 扩展到批次维度
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation # 应用旋转变换
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    #  堆叠并重塑网格以用于grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    #  应用网格采样进行旋转
                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,                   # 采样网格
                        mode="bilinear",        # 双线性插值模式
                        padding_mode="zeros",   # 边界外填充零
                        align_corners=False,    # 不对齐角点
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 对所有相机应用颜色增强
            # 随机亮度调整
            # 使用张量操作而非.item()以保持torch.compile兼容性
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor    # 应用亮度调整

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 随机对比度调整
            # 使用张量操作而非.item()以保持torch.compile兼容性
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)      # 计算图像均值
            image = (image - mean) * contrast_factor + mean     # 应用对比度调整

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 随机饱和度调整（简化版，不使用HSV转换）
            # 使用张量操作而非.item()以保持torch.compile兼容性
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)     # 计算灰度图像
            image = gray + (image - gray) * saturation_factor   # 应用饱和度调整

            # Clamp values to [0, 1] # 将值限制在[0, 1]范围内
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1] # 将值范围从[0, 1]转回[-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        # 如果原始图像是通道优先格式，将处理后的图像转换回该格式
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # 将处理后的图像存入输出字典
        out_images[key] = image

    # obtain 
    # 获取图像掩码
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            # 默认不使用掩码，创建全1掩码
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            # 使用提供的掩码
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    # 创建一个简单的对象，包含所需属性，而不使用复杂的Observation类
    class SimpleProcessedObservation:
        """
        简化的处理后观测数据类，用于存储预处理后的数据。
        该类通过动态设置属性来存储所有传入的数据，避免使用复杂的Observation类。
        参数:
            **kwargs: 关键字参数，将被设置为类的属性
        """
        def __init__(self, **kwargs):
            # 将所有关键字参数设置为类的属性
            for key, value in kwargs.items():
                setattr(self, key, value)

    # 返回包含所有处理后数据的简化观测对象
    return SimpleProcessedObservation(
        images=out_images,                              # 处理后的图像字典
        image_masks=out_masks,                          # 图像掩码字典
        state=observation.state,                        # 原始状态数据
        tokenized_prompt=observation.tokenized_prompt,  # 分词后的提示
        tokenized_prompt_mask=observation.tokenized_prompt_mask,    # 提示掩码
        token_ar_mask=observation.token_ar_mask,        # 自回归掩码
        token_loss_mask=observation.token_loss_mask,    # 损失掩码
    )
