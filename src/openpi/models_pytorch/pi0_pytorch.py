"""
用途

本脚本实现了PIO（Policy Iteration Zero）模型的PyTorch版本，这是一个基于扩散模型的机器人策略学习框架。
该模型结合了PaliGemma视觉语言模型和Gemma专家模型，用于从图像、语言指令和机器人状态生成机器人动作。
模型采用扩散过程进行训练，通过逐步去噪来生成精确的机器人控制动作。

主要功能：
1. 多模态输入处理：处理图像、文本指令和机器人状态
2. 基于扩散模型的动作生成：通过去噪过程生成精确的机器人动作
3. 支持梯度检查点：优化内存使用，支持大模型训练
4. 两种模型变体：支持PIO和PIO.5两种变体
"""
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    """获取对应设备类型的安全数据类型。
    根据设备类型选择合适的数据类型，特别是处理CPU不支持的数据类型。
    参数:
        target_dtype: 目标数据类型
        device_type: 设备类型（如"cpu"或"cuda"）
    返回:
        适合当前设备的数据类型
    """
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        # CPU不支持bfloat16，使用float32代替
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    """计算标量位置的正弦-余弦位置嵌入向量。
    为时间步创建正弦和余弦位置编码，用于模型中的时间步编码。
    参数:
        time: 形状为(batch_size,)的时间张量
        dimension: 嵌入维度，必须是偶数
        min_period: 最小周期
        max_period: 最大周期
        device: 计算设备
    返回:
        形状为(batch_size, dimension)的位置嵌入张量
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    # 计算外积
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """从Beta分布中采样。
    创建Beta分布并从中采样，用于生成时间步。
    参数:
        alpha: Beta分布的alpha参数
        beta: Beta分布的beta参数
        bsize: 批次大小
        device: 计算设备
    返回:
        形状为(bsize,)的Beta分布样本
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    """从填充掩码和注意力掩码创建二维注意力掩码，用于Transformer的注意力机制。
    复制自big_vision库。
    令牌可以关注具有小于或等于其累积掩码的有效输入令牌。这样，`mask_ar` int[B, N] 可用于设置几种类型的注意力，例如：
    [1 1 1 1 1 1]: 纯因果注意力。
    [0 0 0 1 1 1]: 前缀-1m注意力。前3个令牌可以相互关注，后3个令牌有因果注意力。
        第一个条目也可以是1，而不改变行为。
    [1 1 0 1 0 1 0 1 0 0]: 4个块之间的因果注意力。一个块的令牌可以关注所有先前的块和同一块上的所有令牌。
    参数:
        pad_masks: 布尔型[B, N]，如果是输入的一部分则为true，如果是填充则为false
        att_masks: 整型[B, N]，在前一个令牌不能依赖它的地方为1，在与前一个令牌共享相同注意力掩码的地方为0
    返回:
        二维注意力掩码
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """PIO PyTorch模型类。
    这是一个基于扩散模型的机器人策略学习框架的PyTorch实现。
    该模型结合了PaliGemma视觉语言模型和Gemma专家模型，用于从图像、语言指令和机器人状态生成机器人动作。
    """
    def __init__(self, config):
        """初始化PIOPytorch模型。
        参数:
            config: 模型配置对象，包含模型参数和配置信息
        """
        super().__init__()
        self.config = config     # 保存配置
        self.pi05 = config.pi05     # 是否使用PIO.5变体

        # 获取PaliGemma和动作专家模型的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 初始化PaliGemma模型和动作专家模型
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # 动作投影层
        # TODO(tzhang):这里的32是怎么来的？
        self.action_in_proj = nn.Linear(32, action_expert_config.width) # 输入动作投影
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)   # 输出动作投影

        # 根据模型变体初始化不同的网络层
        if self.pi05:
            # PIO.5变体使用时间MLP
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # PIO变体使用状态投影和动作时间MLP
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # 设置高精度矩阵乘法
        torch.set_float32_matmul_precision("high")
        # 编译采样动作函数以提高性能
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # 检查transformers_replace是否正确安装
        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        """启用梯度检查点以优化内存使用。
        为模型的各个组件启用梯度检查点，减少内存使用但可能增加计算时间。
        """
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        """禁用梯度检查点。
        为模型的各个组件禁用梯度检查点，恢复正常的内存使用和计算速度。
        """
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        """检查梯度检查点是否已启用。
        返回:
            布尔值，表示梯度检查点是否已启用
        """
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        """应用梯度检查点的辅助方法。
        如果启用了梯度检查点且处于训练模式，则使用torch.utils.checkpoint包装函数调用。
        参数:
            func: 要应用检查点的函数
            *args: 函数的位置参数
            **kwargs: 函数的关键字参数
        返回:
            函数的输出
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        """准备用于transformer的4D注意力掩码的辅助方法。
        将2D注意力掩码转换为4D格式，并应用适当的掩码值。
        参数:
            att_2d_masks: 2D注意力掩码
        返回:
            4D注意力掩码，适用于transformer模型
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        """预处理观察数据的辅助方法。
        处理输入的观察数据，提取图像、掩码、文本令牌和状态信息。
        参数:
            observation: 原始观察数据
            train: 是否处于训练模式
        返回:
            处理后的图像列表、图像掩码列表、语言令牌、语言掩码和状态
        """
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),  # 图像列表
            list(observation.image_masks.values()), # 图像掩码列表
            observation.tokenized_prompt,   # 分词后的提示文本
            observation.tokenized_prompt_mask,  # 提示文本掩码
            observation.state,  # 机器人状态
        )

    def sample_noise(self, shape, device):
        """采样噪声。
        生成标准正态分布的噪声张量。
        参数:
            shape: 噪声张量的形状
            device: 计算设备

        返回:
            噪声张量
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        """使用SgLIP嵌入图像和使用嵌入层嵌入语言令牌，为PaliGemma transformer处理做准备。
        处理输入的图像和语言令牌，生成嵌入表示、填充掩码和注意力掩码。
        参数:
            images: 图像列表
            img_masks: 图像掩码列表
            lang_tokens: 语言令牌
            lang_masks: 语言掩码

        返回:
            嵌入表示、填充掩码和注意力掩码的元组
        """
        embs = []       # 嵌入列表
        pad_masks = []  # 填充掩码列表
        att_masks = []  # 注意力掩码列表

        # Process images
        # 处理图像
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            
            # 应用梯度检查点嵌入图像
            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            # 创建注意力掩码，使图像令牌可以相互关注
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)   # 缩放嵌入

        # 应用梯度检查点嵌入语言
        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        # 图像和语言输入之间的完全注意力
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # 连接所有嵌入和掩码
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        """嵌入状态、噪声动作和时间步，为专家Gemma处理做准备。
        处理输入的状态、噪声动作和时间步，生成嵌入表示、填充掩码、注意力掩码和adaRMS条件。
        参数:
            state: 机器人状态
            noisy_actions: 带噪声的动作
            timestep: 时间步
        返回:
            嵌入表示、填充掩码、注意力掩码和adaRMS条件的元组
        """
        embs = []       # 嵌入表示
        pad_masks = []  # 填充掩码列表
        att_masks = []  # 注意力掩码列表

        if not self.pi05:   # PIO
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state 嵌入状态
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            # 设置注意力掩码，使图像和语言输入不关注状态或动作
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 使用正弦-余弦位置编码嵌入时间步，敏感度范围为[0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        # 使用MLP融合时间步和动作信息
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:   #PI0
            # 扩展时间嵌入并与动作嵌入连接
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            # 应用MLP层
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:   # PI0.5
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb  # adarms条件

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # Concatenate all embeddings and masks
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        """执行完整的训练前向传播并计算损失（batch_size x num_steps x num_motors）。
        处理观察数据和动作，添加噪声，然后通过模型预测去噪方向。
        参数:
            observation: 观察数据，包含图像和文本指令
            actions: 目标动作
            noise: 可选的预定义噪声
            time: 可选的预定义时间步

        返回:
            每个动作维度的均方误差损失
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 根据时间步混合噪声和动作
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions   # 目标去噪方向

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks) # 嵌入前缀（图像和语言）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)  # 嵌入后缀（状态、噪声动作和时间步）
        
         # 确保数据类型匹配模型权重
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 连接掩码
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 创建二维注意力掩码和位置ID
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks 准备4D注意力掩码
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        # 定义前向函数，可应用梯度检查点
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 提取动作部分并转换为float32
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        # 定义动作输出投影函数
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        # 应用梯度检查点到最终动作投影
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 计算预测去噪方向和真实去噪方向之间的MSE损失
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        """执行完整的推理前向传播并计算动作（batch_size x num_steps x num_motors）。
        通过迭代去噪过程从噪声生成动作，使用欧拉方法进行数值积分。
        参数:
            device: 计算设备
            observation: 观察数据，包含图像和文本指令
            noise: 可选的预定义噪声
            num_steps: 去噪步数

        返回:
            生成的动作张量
        """
        bsize = observation.state.shape[0]  # 批次大小
        if noise is None:
            # 如果未提供噪声，则生成标准高斯噪声
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)    # 生成噪声

        # 预处理观测数据
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # 嵌入前缀（图像和语言）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        #  计算图像和语言的键值缓存，以加速推理
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        # 前向传播计算键值缓存
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 计算时间步长
        dt = -1.0 / num_steps   # 时间步长
        dt = torch.tensor(dt, dtype=torch.float32, device=device)   # 转换为tensor
    
         # 从纯噪声开始
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)    # 初始时间
        # 迭代去噪过程，使用欧拉方法
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)  # 扩展时间
            # 计算去噪方向
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            # 欧拉步骤 - 使用新张量赋值而不是就地操作
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        """
        在给定时间同步对噪声 'x_t' 应用一步去噪。
        使用模型预测去噪方向，用于扩散采样过程中的单步去噪。
        参数:
            state: 机器人状态
            prefix_pad_masks: 前缀填充掩码
            past_key_values: 预计算的键值缓存
            x_t: 当前噪声动作
            timestep: 当前时间步
        返回:
            预测的去噪方向
        """
        # 嵌入后缀（状态、噪声动作和时间步）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # 获取维度信息
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 创建前缀填充2D掩码
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # 创建后缀注意力2D掩码
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 连接前缀和后缀掩码
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 计算位置ID，考虑前缀偏移
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        # 准备4D注意力掩码
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        #  前向传播，使用预计算的键值缓存
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # 提取后缀输出并转换为float32
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # 投影到动作维度、并返回
        return self.action_out_proj(suffix_out)
