"""
策略模块 - 将模型封装为可部署的策略接口

本模块提供了策略执行层，将训练好的模型包装为统一的推理接口。

核心类:
1. Policy: 主策略类，封装模型推理和数据变换流程
2. PolicyRecorder: 策略包装器，用于记录推理过程到磁盘

主要功能:
- 统一的推理接口：infer(obs) -> actions
- 数据预处理：自动应用输入变换（归一化、图像调整等）
- 数据后处理：自动应用输出变换（反归一化、动作空间转换等）
- 多框架支持：同时支持JAX和PyTorch模型
- 性能监控：记录推理时间等指标

使用示例:
    # 创建策略
    policy = Policy(
        model=trained_model,
        transforms=[normalize, resize_images],
        output_transforms=[unnormalize, delta_to_absolute],
        sample_kwargs={"num_steps": 10}
    )

    # 推理
    obs = {"image": {...}, "state": ...}
    result = policy.infer(obs)
    actions = result["actions"]
    timing = result["policy_timing"]
"""
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """策略类 - 封装模型为统一的推理接口

    Policy类将训练好的模型包装为标准化的推理接口，处理数据的
    预处理、模型推理和后处理流程。

    主要特性:
    - 自动数据变换：输入和输出的自动转换
    - 框架无关：支持JAX和PyTorch模型
    - 批处理：自动添加/移除批次维度
    - 性能跟踪：记录推理时间
    - 元数据管理：存储策略相关的配置信息

    数据流：
        原始观察 -> 输入变换 -> 模型推理 -> 输出变换 -> 最终动作
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """初始化策略

        参数:
            model: 用于动作采样的模型（JAX或PyTorch）
            rng: JAX模型的随机数生成器密钥，PyTorch模型会忽略
            transforms: 推理前应用的输入数据变换序列
            output_transforms: 推理后应用的输出数据变换序列
            sample_kwargs: 传递给model.sample_actions的额外关键字参数
            metadata: 与策略一起存储的额外元数据
            pytorch_device: PyTorch模型使用的设备（如"cpu", "cuda:0"）
                          仅在is_pytorch=True时相关
            is_pytorch: 模型是否为PyTorch模型。False表示JAX模型

        示例:
            >>> policy = Policy(
            ...     model=pi0_model,
            ...     transforms=[normalize, resize],
            ...     output_transforms=[unnormalize],
            ...     sample_kwargs={"num_steps": 10},
            ...     is_pytorch=False
            ... )
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # PyTorch模型设置
            self._model = self._model.to(pytorch_device)
            self._model.eval()  # 设置为评估模式
            self._sample_actions = model.sample_actions
        else:
            # JAX模型设置
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """执行策略推理，从观察数据生成动作

        参数:
            obs: 观察数据字典，包含 "image", "state" 等键
            noise: 可选的初始噪声（用于扩散模型），形状为(action_horizon, action_dim)
                  或(1, action_horizon, action_dim)

        返回:
            包含以下键的字典:
            - "actions": 预测的动作序列 [action_horizon, action_dim]
            - "state": 当前状态（从输入复制）
            - "policy_timing": 性能指标字典
                - "infer_ms": 推理时间（毫秒）

        注意:
            - 输入会被复制以避免原地修改
            - 自动处理批次维度的添加和移除
            - 自动应用配置的输入和输出变换
        """
        # 复制输入，因为变换可能会原地修改数据
        inputs = jax.tree.map(lambda x: x, obs)
        # 应用输入变换（归一化、图像调整等）
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # JAX模型：转换为JAX数组并添加批次维度
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # PyTorch模型：转换为PyTorch张量并移动到正确设备
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # 准备sample_actions的参数
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            # 转换噪声到正确的框架和设备
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            # 如果噪声是2D，添加批次维度
            if noise.ndim == 2:  # (action_horizon, action_dim)
                noise = noise[None, ...]  # -> (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # 从字典创建Observation对象
        observation = _model.Observation.from_dict(inputs)

        # 执行模型推理并计时
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        # 转换输出为NumPy数组并移除批次维度
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 应用输出变换（反归一化、动作空间转换等）
        outputs = self._output_transform(outputs)

        # 添加性能指标
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """获取策略元数据

        返回:
            包含策略配置和额外信息的字典
        """
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """策略记录器 - 将策略的输入输出记录到磁盘

    这个包装器用于调试和数据收集，记录策略推理过程中的
    所有输入和输出数据。

    用途:
    - 调试策略行为
    - 收集示范数据
    - 性能分析
    - 可视化检查

    记录格式:
        每次推理保存为一个.npy文件，包含扁平化的字典:
        {
            "inputs/image/cam0": ...,
            "inputs/state": ...,
            "outputs/actions": ...,
            ...
        }
    """

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        """初始化策略记录器

        参数:
            policy: 要包装的基础策略
            record_dir: 记录文件保存目录

        注意:
            目录会自动创建（如果不存在）
        """
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        """执行推理并记录输入输出

        参数:
            obs: 观察数据字典

        返回:
            策略的推理结果（与被包装策略相同）

        副作用:
            将输入和输出保存到 record_dir/step_{step_num}.npy
        """
        # 执行实际推理
        results = self._policy.infer(obs)

        # 组合输入和输出数据
        data = {"inputs": obs, "outputs": results}
        # 扁平化嵌套字典以便于保存
        data = flax.traverse_util.flatten_dict(data, sep="/")

        # 保存到文件
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
