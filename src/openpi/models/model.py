"""
模型基础模块 - 提供OpenPI模型的核心接口和数据结构

本模块定义了：
1. 模型类型枚举（ModelType）- 支持的模型变体
2. 观察数据结构（Observation）- 统一的多模态输入格式
3. 动作数据类型（Actions）- 动作序列的标准表示
4. 模型配置基类（BaseModelConfig）- 所有模型的配置接口
5. 模型基类（BaseModel）- 所有模型实现的抽象基类
6. 工具函数：
   - preprocess_observation: 观察数据预处理（图像增强、调整大小等）
   - restore_params: 从检查点恢复模型参数

核心设计原则：
- 统一的数据格式：所有模型使用相同的Observation和Actions数据结构
- 多框架支持：同时支持JAX和PyTorch实现
- 类型安全：使用类型注解和运行时类型检查
- 可扩展性：通过抽象基类支持新模型的添加
"""
import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types.

    支持的模型类型枚举

    - PI0: 基础版本的Pi0模型，使用扩散模型进行动作预测
      Base version of Pi0 model, using diffusion model for action prediction
    - PI0_FAST: Pi0的快速变体，使用自回归token生成
      Fast variant of Pi0, using autoregressive token generation
    - PI05: Pi0.5版本，引入了AdaRMS（自适应RMS归一化）机制
      Pi0.5 version, introducing AdaRMS (Adaptive RMS normalization) mechanism
    """

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
# 模型默认期望的图像输入键名
# 这三个视角分别对应：基座摄像头、左手腕摄像头、右手腕摄像头
# These three views correspond to: base camera, left wrist camera, right wrist camera
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
# 模型输入图像的标准分辨率 (高度, 宽度)
# 注意：如果发布小模型，这个分辨率可能会改变
IMAGE_RESOLUTION = (224, 224)


# Data format
# 数据格式说明
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Observation` and `Actions` objects. See below.
# 数据变换（transforms）产生的模型输入是一个嵌套字典，随后会被转换为 `Observation` 和 `Actions` 对象。
#
# In the dictionary form, this data should look like:
# 在字典形式中，数据结构应该如下所示：
# {
#     # Observation data.
#     # 观察数据部分
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#                                                        # RGB图像，范围在[-1, 1]或[0, 255]
#         ...  # Additional camera views / 其他摄像头视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid / True表示图像有效，False表示填充
#         ...  # Masks for additional views / 其他视角的掩码
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state / 低维机器人状态（关节角度、位置等）
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt / 可选，分词后的语言提示
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt / 可选，提示词的掩码
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model / 可选，FAST模型的自回归掩码
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model / 可选，FAST模型的损失掩码
#
#      # Actions data. / 动作数据部分
#      "actions": float32[*b, ah, ad]  # 动作序列
# }
# where: / 其中：
#   *b = batch dimensions / 批次维度（可能有多个）
#   h, w = image height/width / 图像高度/宽度
#   s = state dimension / 状态维度
#   l = sequence length / 序列长度
#   ah = action_horizon / 动作序列长度
#   ad = action_dim / 动作维度
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    观察数据结构 - 存储模型的所有输入信息

    Observation类封装了机器人的多模态观察数据，包括：
    Observation class encapsulates multi-modal robotic observation data, including:
    - 多视角图像：来自不同摄像头的RGB图像
      Multi-view images: RGB images from different cameras
    - 图像掩码：标识哪些图像视角是有效的
      Image masks: Identify which image views are valid
    - 机器人状态：关节角度、末端执行器位置等低维状态
      Robot state: Low-dimensional state such as joint angles, end-effector positions
    - 语言指令：可选的自然语言任务描述（已分词）
      Language instructions: Optional natural language task descriptions (tokenized)

    数据类型参数 / Type parameter:
        ArrayT: 数组类型，可以是JAX数组、PyTorch张量或NumPy数组
                Array type, can be JAX array, PyTorch tensor, or NumPy array

    使用方法 / Usage:
        1. 从字典创建：Observation.from_dict(data_dict)
           Create from dict: Observation.from_dict(data_dict)
        2. 转换为字典：observation.to_dict()
           Convert to dict: observation.to_dict()

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    参考 `Observation.from_dict` 方法查看预期的字典格式。
    这是数据变换（transforms）应该产生的格式。
    """

    # Images, in [-1, 1] float32.
    # 图像数据，范围在 [-1, 1] 的 float32
    # 键为摄像头名称（如 "base_0_rgb"），值为对应的图像数组
    # Keys are camera names (e.g., "base_0_rgb"), values are corresponding image arrays
    images: dict[str, at.Float[ArrayT, "*b h w c"]]

    # Image masks, with same keys as images.
    # 图像掩码，键与 images 相同
    # True 表示该图像有效，False 表示填充或无效数据
    # True indicates valid image, False indicates padding or invalid data
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]

    # Low-dimensional robot state.
    # 低维机器人状态向量
    # 通常包含关节角度、末端执行器位置等信息
    # Usually contains joint angles, end-effector positions, etc.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    # 分词后的语言提示（可选）
    # 用于语言条件的策略学习
    # For language-conditioned policy learning
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None

    # Tokenized prompt mask.
    # 提示词掩码（可选）
    # 标识提示词序列中哪些token是有效的
    # Identifies which tokens in the prompt sequence are valid
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.
    # pi0-fast 模型特定字段

    # Token auto-regressive mask (for FAST autoregressive model).
    # Token自回归掩码（用于FAST自回归模型）
    # 控制自回归生成过程中的依赖关系
    # Controls dependencies in the autoregressive generation process
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None

    # Token loss mask (for FAST autoregressive model).
    # Token损失掩码（用于FAST自回归模型）
    # 标识在训练时哪些token应该参与损失计算
    # Identifies which tokens should participate in loss calculation during training
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format.

        从嵌套字典创建Observation对象
        此方法定义了非结构化数据（嵌套字典）到结构化Observation格式的映射。

        参数 / Args:
            data: 包含观察数据的嵌套字典，应符合上述数据格式说明
                  Nested dictionary containing observation data, should conform to the data format description above

        返回 / Returns:
            Observation对象 / Observation object

        注意 / Note:
            - tokenized_prompt 和 tokenized_prompt_mask 必须同时提供或同时缺失
              tokenized_prompt and tokenized_prompt_mask must be provided together
            - uint8类型的图像会自动转换为[-1, 1]范围的float32
              uint8 images are automatically converted to [-1, 1] float32
        """
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict.

        将Observation转换为嵌套字典

        返回 / Returns:
            包含所有观察数据的嵌套字典
            Nested dictionary containing all observation data
        """
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
# 动作数据类型定义
# 表示动作序列：[批次维度, 动作序列长度, 动作维度]
# 这个字段在数据变换产生的字典中以 "actions" 键存储
# Represents action sequence: [batch dimensions, action_horizon, action_dim]
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """
    
    """预处理观察数据，包括图像增强、调整大小和掩码处理
        参数:
            rng (JAX随机数生成器): 用于图像增强时的随机变换。在推理模式下可为None。
            observation (原始观察数据): 包含图像、状态等信息。
            train (是否为训练模式): 影响是否进行图像增强。训练时启用增强以提高泛化能力。
            image_keys (需要处理的图像键名列表): 默认包含三个视角的图像。
            image_resolution (目标图像分辨率): 模型要求固定大小的输入。

        返回:
            预处理后的观察数据，包含统一处理后的图像和适当的掩码。

        预处理步骤及其重要性：
        1. 图像大小调整：
        - 目的：确保所有图像具有相同的尺寸
        - 方法：使用带填充的调整大小方法，保持图像原始宽高比
        - 意义：允许批处理加速计算，提供一致的视觉信号给模型

        2. 图像增强（仅在训练模式）：
        - 目的：增加训练数据的多样性，提高模型泛化能力，防止过拟合
        - 方法：根据图像类型应用不同的增强策略：
            a) 基础视角图像：空间变换（裁剪、缩放、旋转）和颜色变换
            b) 手腕视角图像：仅颜色变换（保持空间结构不变）
        - 意义：模拟现实环境中的变化，使模型更加鲁棒，能应对不同光照和视角

        3. 掩码处理：
        - 目的：为每个图像提供有效性标记，告知模型哪些视图的数据可用
        - 方法：使用已有掩码或创建默认全有效掩码
        - 意义：在多视角融合时提供权重依据，使模型能处理部分缺失的观察数据
        """

    # 处理前确保不会因缺少必要的视觉数据而失败
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 批次维度通常是第一维，保留它可以支持批量处理多个样本
    batch_shape = observation.state.shape[:-1]

    # 处理每个图像
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        # 如果图像尺寸与目标分辨率不同，进行调整，因为神经网络需要固定大小的输入
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        # 图像增强步骤（仅在训练模式下进行）
        # 目的是增加训练数据的多样性，提高模型的泛化能力
        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            # 将范围从[-1, 1]范围转换到[0, 1]范围，以便进行增强
            image = image / 2.0 + 0.5

            transforms = []
            # 对于非手部视角的图像，添加额外的空间变换
            # 手腕视角图像不进行空间变换的原因可能是：
            # 1. 手腕视角对空间位置信息更敏感
            # 2. 保持手部动作的精确空间关系对任务执行很重要
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    # 随机裁剪到原尺寸的95%，然后缩放回原尺寸
                    # 这模拟了不同的视野范围，使模型对物体位置变化更鲁棒
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    # 随机旋转±5度，这模拟了相机角度的微小变化，提高模型对视角变化的适应性
                    augmax.Rotate((-5, 5)),
                ]
            # 对所有图像进行颜色增强，提高对于光照变化的鲁棒性
            transforms += [
                # 调整亮度、对比度和饱和度
                # 增强参数(0.3, 0.4, 0.5)表示变化的强度，数值越大变化越明显
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            # 为每个批次样本生成独立的随机数，确保批次中的每个样本都有不同的随机增强效果
            sub_rngs = jax.random.split(rng, image.shape[0])
            # jax.vmap将增强函数映射到批次中的每个样本，这是一种并行处理的方式，提高了计算效率
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            # 将图像从[0, 1]范围转换回[-1, 1]范围，因为前面将图像范围从[-1, 1]转到了[0, 1]
            image = image * 2.0 - 1.0
        #  存储处理后的图像
        # 每个视角的图像都经过了一致的预处理
        out_images[key] = image

    # obtain mask
    # 处理图像掩码，掩码标识哪些图像是有效的，对于多视角融合很重要
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            # 如果没有提供掩码，默认所有位置都是有效的（全1掩码），表示所有图像数据都可用
            # 在真实应用中，某些视角可能因遮挡或设备故障而不可用
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            # 使用提供的掩码
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 返回预处理后的观察数据
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """模型配置基类 - 所有模型配置必须继承此类

    此抽象基类定义了所有OpenPI模型共享的配置参数和接口。
    具体模型应该继承此类，并实现 `create` 方法来创建对应的模型实例。

    配置参数:
        action_dim: 动作空间维度（例如：机器人关节数量）
        action_horizon: 动作序列长度（预测未来多少步）
        max_token_len: 分词提示词的最大长度

    必须实现的抽象方法:
        model_type: 返回模型类型枚举
        create: 创建并初始化新模型
        inputs_spec: 返回模型输入规范

    提供的工具方法:
        load: 从参数字典加载模型
        load_pytorch: 加载PyTorch模型
        fake_obs: 生成虚拟观察数据（用于测试）
        fake_act: 生成虚拟动作数据（用于测试）
    """

    # 动作空间维度
    action_dim: int
    # 动作序列长度
    action_horizon: int
    # 分词提示词最大长度
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """返回模型类型枚举"""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建新模型，初始化参数

        参数:
            rng: JAX随机数生成器，用于参数初始化

        返回:
            初始化后的模型实例
        """

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """从参数字典创建模型

        参数:
            params: 模型参数字典（PyTree格式）
            remove_extra_params: 是否移除额外的参数（不在模型中的参数）

        返回:
            加载了指定参数的模型实例

        注意:
            此方法会检查参数的形状是否匹配，但不检查数据类型
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        """加载PyTorch模型

        参数:
            train_config: 训练配置对象
            weight_path: 权重文件路径

        返回:
            加载了权重的PyTorch模型
        """
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型输入规范

        参数:
            batch_size: 批次大小

        返回:
            (observation_spec, action_spec) 元组，值为 jax.ShapeDtypeStruct
        """

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """生成虚拟观察数据（用于测试）

        参数:
            batch_size: 批次大小

        返回:
            填充了全1的观察数据
        """
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        """生成虚拟动作数据（用于测试）

        参数:
            batch_size: 批次大小

        返回:
            填充了全1的动作数据
        """
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """模型基类 - 所有模型实现必须继承此类

    此抽象基类定义了所有OpenPI模型的核心接口。
    具体模型应该继承此类，并实现抽象方法。

    子类必须调用 super().__init__() 来初始化共享属性：
    - action_dim: 动作空间维度
    - action_horizon: 动作序列长度
    - max_token_len: 分词提示词最大长度

    必须实现的抽象方法:
        compute_loss: 计算训练损失
        sample_actions: 从观察数据采样动作序列
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """计算模型损失

        参数:
            rng: JAX随机数生成器
            observation: 观察数据
            actions: 目标动作序列
            train: 是否为训练模式（影响dropout等）

        返回:
            每个样本的损失值，形状为 [*batch_size, action_horizon]
        """
        ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        """从观察数据采样动作序列

        参数:
            rng: JAX随机数生成器（用于采样过程）
            observation: 当前观察数据
            **kwargs: 模型特定的采样参数

        返回:
            采样的动作序列
        """
        ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
