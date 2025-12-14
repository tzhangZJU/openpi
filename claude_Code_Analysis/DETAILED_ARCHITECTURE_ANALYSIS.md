# OpenPI 深度架构分析与解读
# OpenPI Deep Architecture Analysis and Interpretation

**生成日期 | Date**: 2025-12-14
**分析重点 | Focus**: 模型架构、Flow Matching、Joint Attention、Fast Tokenizer

---

## 目录 | Table of Contents

1. [模型架构总览](#模型架构总览)
2. [Pi0 模型详解](#pi0-模型详解)
3. [Pi0-Fast 模型详解](#pi0-fast-模型详解)
4. [Pi0.5 模型详解](#pi05-模型详解)
5. [Flow Matching 深度解析](#flow-matching-深度解析)
6. [Joint Attention 机制](#joint-attention-机制)
7. [Fast Tokenizer 实现](#fast-tokenizer-实现)

---

## 模型架构总览

### 三种模型变体对比

| 特性 | Pi0 | Pi0-Fast | Pi0.5 |
|------|-----|----------|-------|
| **动作生成方式** | 扩散模型 (Diffusion) | 自回归 Token 生成 | 扩散模型 + AdaRMS |
| **推理速度** | 慢 (需要多步ODE求解) | 快 (单次前向传播) | 慢 (多步ODE求解) |
| **时间步嵌入** | MLP混合 | 不需要 | AdaRMS条件化 |
| **状态处理** | 独立状态token | 融入提示词 | 无独立状态token |
| **适用场景** | 高精度任务 | 实时控制 | 高精度 + 更好泛化 |

### 共享组件

所有三个模型变体都共享以下核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                    视觉编码器 Vision Encoder                 │
│                    SigLIP (So400m/14)                       │
│                                                             │
│  输入: RGB图像 [B, 3, 224, 224]                             │
│  输出: 视觉token [B, 256, 1152]                             │
│       (3个视角 × 256 tokens/view)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 语言-视觉模型 Vision-Language Model          │
│                    PaliGemma (2B params)                    │
│                                                             │
│  功能: 融合视觉特征和语言指令                                │
│  结构: Transformer (多头注意力 + FFN)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  动作专家 Action Expert                      │
│                    Gemma-based (参数独立)                    │
│                                                             │
│  功能: 预测动作序列                                          │
│  输入: 视觉-语言特征 + 状态/动作信息                          │
│  输出: 动作预测 [B, action_horizon, action_dim]             │
└─────────────────────────────────────────────────────────────┘
```

---

## Pi0 模型详解

### 整体架构图

```
输入观察 Observation
│
├── 图像 Images (3 views × [224, 224, 3])
│   │
│   └──> SigLIP Encoder
│        └──> 视觉tokens [B, 768, 1152]
│             (3 views × 256 tokens)
│
├── 语言提示 Language Prompt (optional)
│   │
│   └──> Tokenizer
│        └──> 文本tokens [B, L, 1152]
│
├── 机器人状态 Robot State [B, state_dim]
│   │
│   └──> Linear Projection
│        └──> 状态token [B, 1, 1152]
│
└── 噪声动作 Noisy Actions [B, H, action_dim]
    │
    └──> Linear Projection
         └──> 动作tokens [B, H, 1152]

                    ↓ 拼接所有tokens

┌─────────────────────────────────────────────────────────┐
│              Prefix Tokens (前缀部分)                    │
│  [视觉tokens | 文本tokens]                               │
│  注意力模式: 全连接 (bidirectional)                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              Suffix Tokens (后缀部分)                    │
│  [状态token | 动作tokens]                                │
│  注意力模式: 因果 (causal)                               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│                PaliGemma Transformer                    │
│                                                         │
│  Layer 1: Multi-Head Attention + FFN                   │
│  Layer 2: Multi-Head Attention + FFN                   │
│  ...                                                    │
│  Layer N: Multi-Head Attention + FFN                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              Action Expert (Gemma)                      │
│                                                         │
│  输入: 所有tokens的上下文表示                            │
│  处理: 提取动作tokens的表示                              │
│  输出: 速度场预测 v_t [B, H, action_dim]                │
└─────────────────────────────────────────────────────────┘
                    ↓
              速度场 v_t
              (用于Flow Matching)
```

### 前向传播详细流程

#### 1. 前缀嵌入 (Prefix Embedding)

```python
def embed_prefix(self, obs: Observation):
    """
    将观察数据的视觉和语言部分转换为token序列

    处理流程:
    1. 视觉编码
       - 输入: 3个视角的RGB图像，每个 [B, 224, 224, 3]
       - SigLIP处理: 每个视角 → 256个tokens
       - 输出: [B, 768, 1152] (768 = 3 × 256)

    2. 语言编码 (可选)
       - 输入: 分词后的文本 [B, L]
       - PaliGemma嵌入: token_id → 嵌入向量
       - 输出: [B, L, 1152]

    3. 掩码生成
       - image_mask: 标识哪些视角有效
       - ar_mask: 全False (前缀内部全连接注意力)
    """
    tokens = []
    input_mask = []
    ar_mask = []

    # 处理每个视角的图像
    for name in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        image = obs.images[name]  # [B, 224, 224, 3]

        # SigLIP编码: 图像 → 视觉tokens
        # SigLIP内部流程:
        # 1. Patch Embedding: 将224×224图像分成14×14个patch
        # 2. Linear Projection: 每个patch → 1152维向量
        # 3. Vision Transformer: 16层transformer处理
        image_tokens, _ = self.PaliGemma.img(image, train=False)
        # image_tokens shape: [B, 256, 1152]

        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=256)
        )
        # ar_mask为False表示前缀内部可以互相attend
        ar_mask += [False] * 256

    # 处理语言提示 (如果存在)
    if obs.tokenized_prompt is not None:
        # PaliGemma的嵌入层: token_id → 向量
        text_tokens = self.PaliGemma.llm(
            obs.tokenized_prompt, method="embed"
        )  # [B, L, 1152]

        tokens.append(text_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * text_tokens.shape[1]

    # 拼接所有前缀tokens
    tokens = jnp.concatenate(tokens, axis=1)
    # 最终shape: [B, 768+L, 1152] 或 [B, 768, 1152] (无语言时)

    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)

    return tokens, input_mask, ar_mask
```

#### 2. 后缀嵌入 (Suffix Embedding)

```python
def embed_suffix(self, obs: Observation, noisy_actions: Actions, timestep: float):
    """
    处理状态和带噪声的动作，为扩散过程准备

    关键设计:
    1. 状态token作为桥梁连接前缀和动作
    2. 时间步通过正弦余弦编码嵌入
    3. 时间信息与动作信息通过MLP融合
    """
    tokens = []
    input_mask = []
    ar_mask = []

    # ==================== 状态处理 ====================
    # 状态向量投影为token
    state_token = self.state_proj(obs.state)[:, None, :]
    # shape: [B, 1, 1152]

    tokens.append(state_token)
    input_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
    # ar_mask为True: 前缀不能attend到状态
    ar_mask += [True]

    # ==================== 动作处理 ====================
    # 动作投影
    action_tokens = self.action_in_proj(noisy_actions)
    # shape: [B, action_horizon, 1152]

    # ==================== 时间步嵌入 ====================
    # 正弦余弦位置编码
    time_emb = posemb_sincos(
        timestep,
        embed_dim=1152,
        min_period=4e-3,  # 高频分量
        max_period=4.0    # 低频分量
    )  # shape: [B, 1152]

    # 时间步嵌入的原理:
    # 使用不同频率的正弦余弦波编码时间信息
    # freq_i = 1 / (min_period * (max_period/min_period)^(i/d))
    # emb = [sin(t*freq_0), ..., sin(t*freq_{d/2}),
    #        cos(t*freq_0), ..., cos(t*freq_{d/2})]

    # ==================== MLP混合时间和动作 ====================
    # 扩展时间嵌入到动作序列长度
    time_tokens = einops.repeat(
        time_emb, "b emb -> b h emb", h=action_horizon
    )  # [B, action_horizon, 1152]

    # 拼接动作和时间
    action_time_tokens = jnp.concatenate(
        [action_tokens, time_tokens], axis=-1
    )  # [B, action_horizon, 2304]

    # MLP融合
    # 这个MLP的作用是学习如何将时间信息和动作信息结合
    action_time_tokens = self.action_time_mlp_in(action_time_tokens)
    action_time_tokens = nnx.swish(action_time_tokens)
    action_time_tokens = self.action_time_mlp_out(action_time_tokens)
    # shape: [B, action_horizon, 1152]

    tokens.append(action_time_tokens)
    input_mask.append(jnp.ones((B, action_horizon), dtype=jnp.bool_))
    # 第一个动作token的ar_mask为True (前面的不能attend到它)
    # 后续为False (同一序列内因果注意力)
    ar_mask += [True] + [False] * (action_horizon - 1)

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)

    return tokens, input_mask, ar_mask, None  # Pi0不使用adarms_cond
```

#### 3. 注意力掩码构建

```python
def make_attn_mask(input_mask, mask_ar):
    """
    构建注意力掩码矩阵

    原理:
    通过累积和实现灵活的注意力模式

    示例:
    假设序列为 [img1, img2, img3, state, act1, act2]
    ar_mask =    [F,    F,    F,    T,     T,    F   ]

    累积和:      [0,    0,    0,    1,     2,    2   ]

    注意力矩阵 (i可以attend到j当且仅当 cumsum[i] >= cumsum[j]):
              img1  img2  img3  state act1  act2
    img1   [  T     T     T     F     F     F   ]
    img2   [  T     T     T     F     F     F   ]
    img3   [  T     T     T     F     F     F   ]
    state  [  T     T     T     T     F     F   ]
    act1   [  T     T     T     T     T     F   ]
    act2   [  T     T     T     T     T     T   ]

    解释:
    - 图像tokens之间全连接 (可以互相看到)
    - 状态可以看到图像，但图像看不到状态
    - 动作之间因果 (只能看到之前的)
    - 动作可以看到前缀和状态
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)

    # 注意力掩码: cumsum[i] >= cumsum[j] 表示i可以attend到j
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

    # 同时考虑输入掩码 (padding)
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    return jnp.logical_and(attn_mask, valid_mask)
```

### 训练过程详解

见后续 [Flow Matching 深度解析](#flow-matching-深度解析) 章节。

---

## Pi0-Fast 模型详解

### 核心差异

Pi0-Fast 将扩散模型改为**自回归token生成**，类似于语言模型。

### 架构图

```
输入观察 Observation
│
├── 视觉 + 语言前缀 (与Pi0相同)
│   └──> [B, prefix_len, 1152]
│
└── 动作序列 (作为tokens)
    │
    └──> Tokenizer (将连续动作离散化)
         └──> 动作token_ids [B, action_horizon]

                ↓

┌──────────────────────────────────────────────────┐
│            自回归生成 Autoregressive               │
│                                                  │
│  t=0: 输入前缀 → 预测 action_token[0]            │
│  t=1: 输入前缀+action[0] → 预测 action_token[1]  │
│  ...                                             │
│  t=H-1: 输入前缀+action[0:H-1] → 预测 action[H-1]│
└──────────────────────────────────────────────────┘
                ↓
        Detokenizer (离散→连续)
                ↓
    连续动作序列 [B, H, action_dim]
```

### Fast Tokenizer 实现

```python
class FastTokenizer:
    """
    将连续动作向量离散化为tokens

    方法:
    使用码本 (codebook) 进行向量量化 (Vector Quantization)
    类似于VQ-VAE的思想
    """

    def __init__(self, action_dim, codebook_size=1024):
        """
        参数:
            action_dim: 动作向量维度
            codebook_size: 码本大小 (可用token数量)
        """
        # 码本: 预定义的动作向量集合
        # shape: [codebook_size, action_dim]
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, action_dim)
        )

    def encode(self, actions):
        """
        连续动作 → 离散token

        方法: 最近邻搜索
        对于每个动作向量，找到码本中最接近的向量

        参数:
            actions: [B, H, action_dim]
        返回:
            token_ids: [B, H]
        """
        B, H, D = actions.shape

        # 展平为 [B*H, D]
        actions_flat = actions.reshape(-1, D)

        # 计算与码本中每个向量的距离
        # [B*H, D] @ [D, codebook_size] = [B*H, codebook_size]
        distances = torch.cdist(
            actions_flat, self.codebook, p=2
        )

        # 找到最近的码本索引
        token_ids = torch.argmin(distances, dim=-1)
        # shape: [B*H]

        return token_ids.reshape(B, H)

    def decode(self, token_ids):
        """
        离散token → 连续动作

        参数:
            token_ids: [B, H]
        返回:
            actions: [B, H, action_dim]
        """
        # 直接查表
        return self.codebook[token_ids]

    def forward_with_loss(self, actions):
        """
        训练时使用，包含commitment loss

        Commitment Loss的作用:
        鼓励编码器输出接近码本向量，同时更新码本

        loss = ||sg[z_e] - e||^2 + β||z_e - sg[e]||^2
        其中:
            z_e: 编码器输出
            e: 最近的码本向量
            sg: stop gradient
            β: commitment系数
        """
        token_ids = self.encode(actions)
        quantized = self.decode(token_ids)

        # Straight-through estimator
        # 前向: 使用量化后的值
        # 反向: 梯度直通到原始动作
        quantized = actions + (quantized - actions).detach()

        # Commitment loss
        commit_loss = F.mse_loss(actions, quantized.detach())

        return quantized, token_ids, commit_loss
```

### 自回归训练

```python
def compute_loss_fast(self, obs, actions):
    """
    Pi0-Fast的训练损失

    使用teacher forcing进行自回归训练
    """
    # 1. Tokenize动作
    action_tokens = self.tokenizer.encode(actions)
    # shape: [B, H]

    # 2. 嵌入前缀
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)

    # 3. 嵌入动作tokens
    action_embeddings = self.action_embedding(action_tokens)
    # shape: [B, H, 1152]

    # 4. 拼接
    all_tokens = jnp.concatenate([prefix_tokens, action_embeddings], axis=1)

    # 5. 构建因果掩码
    # 前缀内部全连接，动作部分因果
    ar_mask = prefix_ar_mask + [True] + [False] * (H - 1)
    attn_mask = make_attn_mask(all_mask, ar_mask)

    # 6. Transformer前向
    logits = self.llm(all_tokens, mask=attn_mask)
    # shape: [B, prefix_len + H, codebook_size]

    # 7. 提取动作预测部分
    action_logits = logits[:, prefix_len:]
    # shape: [B, H, codebook_size]

    # 8. 计算交叉熵损失
    # 预测: action_logits[:, :-1]  (预测下一个token)
    # 目标: action_tokens[:, 1:]   (实际的下一个token)
    loss = F.cross_entropy(
        action_logits[:, :-1].reshape(-1, codebook_size),
        action_tokens[:, 1:].reshape(-1)
    )

    return loss
```

### 自回归推理

```python
def sample_actions_fast(self, obs, temperature=1.0):
    """
    自回归生成动作序列

    使用KV cache优化效率
    """
    # 1. 嵌入并缓存前缀
    prefix_tokens, prefix_mask, _ = self.embed_prefix(obs)
    _, kv_cache = self.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask
    )

    # 2. 初始化动作序列
    generated_tokens = []

    # 3. 逐步生成
    for t in range(action_horizon):
        if t == 0:
            # 第一步: 只用前缀
            input_tokens = None
        else:
            # 后续步骤: 用上一步的输出
            last_token = generated_tokens[-1]
            input_tokens = self.action_embedding(last_token)[None, None, :]

        # 前向传播 (利用KV cache)
        logits, kv_cache = self.llm(
            [None, input_tokens],
            mask=suffix_attn_mask,
            kv_cache=kv_cache
        )

        # 采样下一个token
        # Temperature scaling控制随机性
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token)

    # 4. 解码为连续动作
    token_ids = torch.cat(generated_tokens, dim=1)
    actions = self.tokenizer.decode(token_ids)

    return actions
```

---

## Pi0.5 模型详解

### 核心创新: AdaRMS

Pi0.5 的主要改进是引入了 **AdaRMS (Adaptive RMS Normalization)**，这是一种条件归一化机制。

### AdaRMS 原理

传统的RMSNorm:
```
RMSNorm(x) = x / RMS(x) * γ

其中 RMS(x) = sqrt(mean(x^2))
     γ: 可学习的缩放参数
```

AdaRMS:
```
AdaRMS(x, cond) = x / RMS(x) * (γ + α(cond))

其中 α(cond): 从条件信息学习的自适应缩放
```

### 架构改进

```
时间步 timestep
    ↓
┌────────────────────────┐
│      Time MLP          │
│   Linear → Swish       │
│   Linear → Swish       │
└────────────────────────┘
    ↓
  time_emb [B, 1152]
    ↓
┌────────────────────────────────────────────┐
│         Action Expert (with AdaRMS)        │
│                                            │
│  Layer 1:                                  │
│    AdaRMS(x, time_emb)                    │
│    Multi-Head Attention                    │
│    AdaRMS(x, time_emb)                    │
│    FFN                                     │
│                                            │
│  Layer 2:                                  │
│    AdaRMS(x, time_emb)                    │
│    Multi-Head Attention                    │
│    AdaRMS(x, time_emb)                    │
│    FFN                                     │
│  ...                                       │
└────────────────────────────────────────────┘
```

### AdaRMS 实现

```python
class AdaRMSNorm(nn.Module):
    """
    自适应RMS归一化

    相比标准RMSNorm的优势:
    1. 可以根据条件信息动态调整归一化强度
    2. 在扩散模型中，时间步信息可以更好地融入网络
    3. 改善梯度流动
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        # 标准缩放参数
        self.scale = nn.Parameter(torch.ones(dim))

        # 条件自适应参数生成器
        # 从条件向量生成额外的缩放因子
        self.cond_proj = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        """
        参数:
            x: [B, L, D] 输入特征
            cond: [B, cond_dim] 条件向量

        返回:
            归一化后的特征 [B, L, D]
        """
        # 1. 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)

        # 2. 归一化
        x_norm = x / rms

        # 3. 生成自适应缩放
        # cond_proj学习如何将时间步信息转换为缩放参数
        adaptive_scale = self.cond_proj(cond)[:, None, :]
        # shape: [B, 1, D]

        # 4. 应用缩放
        # 标准缩放 + 自适应缩放
        output = x_norm * (self.scale + adaptive_scale)

        return output
```

### Pi0.5 后缀嵌入的差异

```python
def embed_suffix_pi05(self, obs, noisy_actions, timestep):
    """
    Pi0.5不使用独立的状态token
    时间信息通过AdaRMS注入到网络中
    """
    tokens = []
    input_mask = []
    ar_mask = []

    # ==================== 动作嵌入 ====================
    action_tokens = self.action_in_proj(noisy_actions)
    # shape: [B, action_horizon, 1152]

    # ==================== 时间嵌入 (用于AdaRMS) ====================
    # 时间嵌入通过MLP处理
    time_emb = posemb_sincos(timestep, 1152, 4e-3, 4.0)
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    # shape: [B, 1152]

    # 注意: 动作tokens不再与时间混合
    # 时间信息将在Transformer层中通过AdaRMS注入
    tokens.append(action_tokens)
    input_mask.append(jnp.ones((B, action_horizon), dtype=jnp.bool_))
    ar_mask += [True] + [False] * (action_horizon - 1)

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)

    # 返回time_emb作为AdaRMS的条件
    return tokens, input_mask, ar_mask, time_emb
```

### Pi0.5 的 Gemma 层 (带AdaRMS)

```python
class GemmaLayerWithAdaRMS(nn.Module):
    """
    Gemma Transformer层，集成AdaRMS
    """

    def __init__(self, config):
        super().__init__()
        self.attn_norm = AdaRMSNorm(config.dim, config.dim)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = AdaRMSNorm(config.dim, config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x, mask, adarms_cond=None):
        """
        参数:
            x: [B, L, D]
            mask: [B, L, L]
            adarms_cond: [B, D] 时间嵌入
        """
        # Pre-norm with AdaRMS
        if adarms_cond is not None:
            x_norm = self.attn_norm(x, adarms_cond)
        else:
            # 如果没有条件，退化为标准RMSNorm
            x_norm = self.attn_norm.standard_norm(x)

        # Attention
        attn_out = self.attention(x_norm, mask=mask)
        x = x + attn_out

        # FFN with AdaRMS pre-norm
        if adarms_cond is not None:
            x_norm = self.ffn_norm(x, adarms_cond)
        else:
            x_norm = self.ffn_norm.standard_norm(x)

        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
```

### Pi0 vs Pi0.5 对比

| 组件 | Pi0 | Pi0.5 |
|------|-----|-------|
| **时间嵌入位置** | 与动作MLP混合 | 通过AdaRMS注入到每层 |
| **状态token** | 有独立的状态token | 无，状态融入前缀 |
| **归一化** | 标准RMSNorm | AdaRMS |
| **参数效率** | 需要time_mlp | time_mlp用于生成条件 |
| **表现** | 基线 | 更好的泛化能力 |

---

## Flow Matching 深度解析

### 理论基础

Flow Matching 是一种连续归一化流 (Continuous Normalizing Flow) 的训练方法，比传统扩散模型更简单高效。

#### 核心思想

将数据分布 $p_{\text{data}}$ 和噪声分布 $p_{\text{noise}}$ 通过一个时间依赖的向量场 (velocity field) 连接起来。

```
t=0: 噪声分布 ε ~ N(0, I)
  ↓
  | 通过ODE: dx/dt = v_θ(x, t)
  ↓
t=1: 数据分布 x ~ p_data
```

#### 数学表述

**概率路径 (Probability Path)**:
```
p_t(x) = (1-t) · p_0(x) + t · p_1(x)

其中:
  p_0 = 噪声分布 N(0, I)
  p_1 = 数据分布 p_data
  t ∈ [0, 1]
```

**条件流 (Conditional Flow)**:
```
给定数据点 x_1 ~ p_data 和噪声 ε ~ N(0, I)
构造路径: x_t = t · x_1 + (1-t) · ε

速度场: v_t = dx_t/dt = x_1 - ε
```

**训练目标**:
```
学习速度场 v_θ(x_t, t) ≈ x_1 - ε

损失函数:
L = E_{x_1, ε, t} [||v_θ(x_t, t) - (x_1 - ε)||²]
```

### Pi0 中的 Flow Matching 实现

#### 训练阶段

```python
def compute_loss(self, rng, observation, actions, train=True):
    """
    Flow Matching 训练损失

    步骤:
    1. 采样时间步 t ~ Beta(1.5, 1)
    2. 采样噪声 ε ~ N(0, I)
    3. 构造噪声动作 x_t = t·actions + (1-t)·ε
    4. 计算目标速度 u_t = ε - actions
    5. 预测速度 v_t = model(obs, x_t, t)
    6. 计算MSE损失
    """

    # ============ 1. 随机数分配 ============
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

    # ============ 2. 观察预处理 ============
    observation = preprocess_observation(
        preprocess_rng, observation, train=train
    )
    # 包括: 图像增强、调整大小等

    # ============ 3. 采样噪声 ============
    batch_shape = actions.shape[:-2]  # [B]
    noise = jax.random.normal(noise_rng, actions.shape)
    # shape: [B, action_horizon, action_dim]

    # ============ 4. 采样时间步 ============
    # Beta(1.5, 1) 分布偏向于较大的t值
    # 这使得训练更关注接近数据的区域
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    # shape: [B]
    # 范围: [0.001, 1.0]

    time_expanded = time[..., None, None]
    # shape: [B, 1, 1] 用于广播

    # ============ 5. 构造噪声样本 ============
    # 线性插值: x_t = t·ε + (1-t)·x_1
    # 注意: 论文中可能写作 x_t = (1-t)·ε + t·x_1
    # 这里使用的约定是 t=1 为噪声，t=0 为数据
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    # shape: [B, action_horizon, action_dim]

    # 为什么这样插值?
    # - t=1: x_t = noise (纯噪声)
    # - t=0: x_t = actions (真实数据)
    # - 0<t<1: x_t 是噪声和数据的混合

    # ============ 6. 计算目标速度场 ============
    # 速度场 v = dx/dt
    # 对于线性路径: v = d/dt[t·ε + (1-t)·x] = ε - x
    u_t = noise - actions
    # shape: [B, action_horizon, action_dim]

    # 理解 u_t:
    # u_t 表示从当前位置指向噪声的方向
    # 在采样时，我们会沿着 -u_t 的方向走 (去噪)

    # ============ 7. 嵌入和拼接tokens ============
    # 前缀: 视觉 + 语言
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
        observation
    )

    # 后缀: 状态 + 噪声动作 + 时间
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
        self.embed_suffix(observation, x_t, time)

    # 拼接
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

    # ============ 8. 构造注意力掩码 ============
    attn_mask = make_attn_mask(input_mask, ar_mask)
    # shape: [B, total_len, total_len]

    positions = jnp.cumsum(input_mask, axis=1) - 1
    # shape: [B, total_len]

    # ============ 9. Transformer前向传播 ============
    # PaliGemma处理前缀
    # Action Expert处理后缀
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond]  # Pi0.5专用
    )
    # suffix_out shape: [B, suffix_len, 1152]

    # ============ 10. 提取动作预测 ============
    # 只需要动作部分 (后缀的最后action_horizon个tokens)
    action_features = suffix_out[:, -self.action_horizon:]
    # shape: [B, action_horizon, 1152]

    # 投影到动作空间
    v_t = self.action_out_proj(action_features)
    # shape: [B, action_horizon, action_dim]

    # ============ 11. 计算损失 ============
    # MSE损失: 预测速度 vs 真实速度
    loss = jnp.square(v_t - u_t)
    # shape: [B, action_horizon, action_dim]

    # 在动作维度上平均
    loss = jnp.mean(loss, axis=-1)
    # shape: [B, action_horizon]

    return loss
```

#### 推理阶段 (ODE采样)

```python
def sample_actions(self, rng, observation, num_steps=10, noise=None):
    """
    使用ODE求解器从噪声生成动作

    核心思想:
    解常微分方程: dx/dt = v_θ(x, t)
    从 t=1 (噪声) 积分到 t=0 (数据)

    数值方法: Euler方法
    """

    # ============ 1. 预处理 ============
    observation = preprocess_observation(None, observation, train=False)

    # ============ 2. 初始化 ============
    # 时间步长 (负数因为从1到0)
    dt = -1.0 / num_steps

    batch_size = observation.state.shape[0]

    # 初始噪声 (t=1)
    if noise is None:
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )
    # shape: [B, action_horizon, action_dim]

    # ============ 3. 缓存前缀 ============
    # 前缀在整个采样过程中不变，可以预先计算并缓存KV
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
        observation
    )

    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    # 前向传播前缀，缓存KV
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions
    )

    # ============ 4. ODE求解 - 定义单步函数 ============
    def step(carry):
        """
        Euler方法的一步

        x_{t+dt} = x_t + dt * v_θ(x_t, t)

        参数:
            carry: (x_t, time)
        返回:
            (x_{t+dt}, time+dt)
        """
        x_t, time = carry

        # 嵌入后缀 (状态 + 当前噪声动作 + 时间)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
            self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size)
            )

        # 构造注意力掩码
        # 后缀内部的因果掩码
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

        # 后缀可以attend到前缀的掩码
        prefix_attn_mask_for_suffix = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )

        # 完整掩码: [前缀 | 后缀]
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask_for_suffix, suffix_attn_mask],
            axis=-1
        )
        # shape: [B, suffix_len, prefix_len + suffix_len]

        # 计算后缀的位置索引
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None] +
            jnp.cumsum(suffix_mask, axis=-1) - 1
        )

        # 前向传播 (使用缓存的前缀KV)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # 前缀已缓存，传None
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond]
        )

        assert prefix_out is None

        # 提取动作特征并投影
        v_t = self.action_out_proj(
            suffix_out[:, -self.action_horizon:]
        )
        # shape: [B, action_horizon, action_dim]

        # Euler更新
        # x_{t+dt} = x_t + dt * v_t
        x_next = x_t + dt * v_t
        time_next = time + dt

        return x_next, time_next

    # ============ 5. 循环终止条件 ============
    def cond(carry):
        """
        继续循环当 time >= -dt/2

        -dt/2 是为了处理浮点误差
        确保能够到达 t=0
        """
        x_t, time = carry
        return time >= -dt / 2

    # ============ 6. 执行ODE求解 ============
    # 使用JAX的while_loop进行高效迭代
    # 初始状态: (noise, t=1.0)
    x_0, _ = jax.lax.while_loop(
        cond,
        step,
        (noise, 1.0)
    )
    # x_0 shape: [B, action_horizon, action_dim]

    # ============ 7. 返回结果 ============
    # x_0 是从噪声去噪得到的动作序列
    return x_0
```

### Flow Matching 的优势

**相比DDPM (Denoising Diffusion Probabilistic Models):**

1. **更简单的训练目标**
   - DDPM: 需要预测噪声，涉及复杂的方差调度
   - Flow Matching: 直接学习速度场，目标更直接

2. **更少的采样步数**
   - DDPM: 通常需要50-1000步
   - Flow Matching: 10-20步就能有好效果

3. **更稳定的训练**
   - 不需要仔细调整噪声调度
   - Beta(1.5, 1)的时间分布自然平衡各时间步的学习

4. **理论更优雅**
   - 基于连续归一化流的严格理论
   - 与ODE求解器直接对应

### 时间步采样策略

```python
# 为什么使用 Beta(1.5, 1)?

import numpy as np
import matplotlib.pyplot as plt

# 生成样本
beta_samples = np.random.beta(1.5, 1, 10000) * 0.999 + 0.001

plt.figure(figsize=(10, 4))

# 直方图
plt.subplot(1, 2, 1)
plt.hist(beta_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('t')
plt.ylabel('Density')
plt.title('Beta(1.5, 1) * 0.999 + 0.001')

# 累积分布
plt.subplot(1, 2, 2)
plt.hist(beta_samples, bins=50, density=True, cumulative=True, alpha=0.7)
plt.xlabel('t')
plt.ylabel('CDF')
plt.title('Cumulative Distribution')

plt.tight_layout()
plt.savefig('beta_distribution.png')

# 分析:
# Beta(1.5, 1) 偏向于较大的t值
# 这意味着训练更关注于接近数据的区域 (小噪声)
# 有助于学习数据分布的细节
```

### ODE求解器的选择

Pi0使用最简单的 **Euler方法**:
```
x_{t+dt} = x_t + dt * v_θ(x_t, t)
```

更高级的方法可以提升质量:

**Heun方法 (二阶)**:
```python
def heun_step(x_t, t, dt, model):
    # 预测步
    v_t = model(x_t, t)
    x_pred = x_t + dt * v_t

    # 校正步
    v_pred = model(x_pred, t + dt)
    x_next = x_t + dt * (v_t + v_pred) / 2

    return x_next
```

**RK4方法 (四阶)**:
```python
def rk4_step(x_t, t, dt, model):
    k1 = model(x_t, t)
    k2 = model(x_t + dt/2 * k1, t + dt/2)
    k3 = model(x_t + dt/2 * k2, t + dt/2)
    k4 = model(x_t + dt * k3, t + dt)

    x_next = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
```

**权衡**:
- Euler: 最快，但需要更多步数
- Heun: 2倍计算，但质量更好
- RK4: 4倍计算，质量最好

---

## Joint Attention 机制

### 多模态融合的挑战

机器人策略需要融合:
1. **视觉信息**: 多视角RGB图像
2. **语言信息**: 任务描述
3. **状态信息**: 关节位置、速度
4. **动作信息**: 历史和未来动作

### Pi0的解决方案: 分层注意力

```
层次1: 前缀注意力 (Prefix Attention)
├── 视觉tokens之间: 全连接
├── 语言tokens之间: 全连接
└── 视觉-语言交互: 全连接

层次2: 后缀注意力 (Suffix Attention)
├── 后缀可以attend到前缀: 单向
├── 状态token: 可以看到前缀
└── 动作tokens: 因果注意力 + 可以看到前缀和状态
```

### 注意力模式可视化

假设序列为:
```
[img0, img1, img2, img3, ..., img767, text0, text1, ..., textL, state, act0, act1, ..., actH]
```

掩码矩阵:
```
         img0 img1 ... img767 text0 ... textL state act0 act1 ... actH
img0   [  1    1   ...   1      1   ...   1     0     0    0   ...  0  ]
img1   [  1    1   ...   1      1   ...   1     0     0    0   ...  0  ]
...
img767 [  1    1   ...   1      1   ...   1     0     0    0   ...  0  ]
text0  [  1    1   ...   1      1   ...   1     0     0    0   ...  0  ]
...
textL  [  1    1   ...   1      1   ...   1     0     0    0   ...  0  ]
state  [  1    1   ...   1      1   ...   1     1     0    0   ...  0  ]
act0   [  1    1   ...   1      1   ...   1     1     1    0   ...  0  ]
act1   [  1    1   ...   1      1   ...   1     1     1    1   ...  0  ]
...
actH   [  1    1   ...   1      1   ...   1     1     1    1   ...  1  ]
```

**解读**:
- **前缀块** (img + text): 全1 → 全连接注意力
- **后缀列** (前缀看后缀): 全0 → 前缀看不到后缀
- **后缀行** (后缀看前缀): 全1 → 后缀可以看到前缀
- **后缀块** (后缀内部): 下三角 → 因果注意力

### 实现细节

```python
def make_attn_mask(input_mask, mask_ar):
    """
    生成注意力掩码

    关键思想:
    使用累积和实现灵活的注意力模式

    参数:
        input_mask: [B, L] bool数组
            True表示该位置是有效输入
        mask_ar: [L] bool数组
            True表示该位置开始新的因果块

    返回:
        [B, L, L] bool数组
            attn_mask[b, i, j] = True 表示位置i可以attend到位置j
    """

    # 1. 广播mask_ar到batch维度
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # shape: [B, L]

    # 2. 计算累积和
    # 累积和的作用:
    # - 相同累积值的位置属于同一个"注意力块"
    # - 块内可以互相attend
    # - 只能attend到累积值 <= 自己的位置
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # shape: [B, L]

    # 示例:
    # mask_ar = [F, F, F, T, F, F, T, F]
    # cumsum  = [0, 0, 0, 1, 1, 1, 2, 2]
    #            \_____/  \______/  \___/
    #            块0      块1       块2

    # 3. 构造注意力掩码
    # cumsum[:, i, :] 是位置i的累积值 (广播到所有j)
    # cumsum[:, :, j] 是所有位置的累积值
    # i可以attend到j 当且仅当 cumsum[i] >= cumsum[j]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # shape: [B, L, L]

    # 4. 考虑padding掩码
    # 只有两个位置都是有效输入时才能attend
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # shape: [B, L, L]

    # 5. 组合
    final_mask = jnp.logical_and(attn_mask, valid_mask)

    return final_mask
```

### 为什么这样设计?

**1. 前缀全连接的原因**:
- 视觉tokens需要相互交互以理解场景
- 语言tokens需要相互交互以理解指令
- 视觉和语言需要对齐 (例如: "抓红色杯子")

**2. 前缀-后缀单向的原因**:
- 动作生成依赖于观察
- 但观察不应该依赖于动作 (避免信息泄漏)

**3. 动作因果的原因**:
- 符合自回归生成的直觉
- 每个时间步的动作只依赖于过去

### 多专家架构

Pi0使用两个独立的Transformer:

```python
# PaliGemma: 处理视觉和语言
paligemma_config = get_config("paligemma-3b")

# Action Expert: 处理动作
action_expert_config = get_config("gemma-2b")

# 它们共享注意力计算，但参数独立
llm = Gemma(
    configs=[paligemma_config, action_expert_config],
    ...
)
```

**前向传播**:
```python
# tokens是一个列表: [prefix_tokens, suffix_tokens]
# 分别送入两个专家

(prefix_out, suffix_out), kv_cache = llm(
    [prefix_tokens, suffix_tokens],
    mask=attn_mask,
    positions=positions
)

# prefix_out: PaliGemma的输出 (通常不使用)
# suffix_out: Action Expert的输出 (用于动作预测)
```

**好处**:
1. **专业化**: 每个专家专注于自己的任务
2. **效率**: 参数不完全共享，减少干扰
3. **灵活性**: 可以独立调整每个专家的大小

---

## 总结

### 关键创新点

1. **Flow Matching**
   - 更简单的训练目标
   - 更少的采样步数
   - 更稳定的训练

2. **Joint Attention**
   - 分层注意力模式
   - 多模态融合
   - 多专家架构

3. **AdaRMS (Pi0.5)**
   - 条件归一化
   - 更好地融合时间信息
   - 改善梯度流动

4. **Fast Tokenizer (Pi0-Fast)**
   - 向量量化
   - 自回归生成
   - 实时控制

### 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 高精度操作 (如手术) | Pi0.5 | 最高质量，AdaRMS改善泛化 |
| 实时控制 (如无人机) | Pi0-Fast | 最快速度，单次前向 |
| 平衡场景 (如家务) | Pi0 | 良好的速度-质量平衡 |
| 研究和开发 | Pi0.5 | 最先进的架构 |

### 未来改进方向

1. **更高效的ODE求解器**
   - 自适应步长
   - 高阶方法 (Heun, RK4)
   - 学习的求解器

2. **更好的码本学习** (Pi0-Fast)
   - 层次化码本
   - 产品量化
   - 残差VQ

3. **更强的多模态融合**
   - 触觉输入
   - 深度信息
   - 音频信号

4. **更大的模型规模**
   - 扩展到10B+参数
   - 混合专家 (MoE)
   - 稀疏注意力
