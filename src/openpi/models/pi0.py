import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """
        计算模型的损失函数
        参数:
            rng (at.KeyArrayLike): JAX随机数生成器，用于生成噪声和采样时间同步
            observation (_model.Observation): 环境观察数据，包含图像、状态等信息
            actions (_model.Actions): 动作序列，形状为 [batch_size, action_horizon, action_dim]
            train (bool, optional): 是否为训练模式，默认为False。影响数据预处理和dropout等行为

        返回:
            at.Float[at.Array, "*b ah"]: 每个样本的损失值，形状为 [batch_size, action_horizon]
        实现细节：
        1. 添加噪声到动作序列
        2. 预测噪声
        3. 计算MSE损失
        """
        # 将随机数生成器分成三份，分别用于预处理、生成噪声和采样时间同步
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 预处理观察数据（图像、状态等）
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]    # 获取batch_size，排除最后两个维度action_horizon和action_dim
        noise = jax.random.normal(noise_rng, actions.shape) # 生成与动作序列相同形状的高斯噪声
        
        # 使用beta分布采样时间步，范围在0.001到1之间
        # beta（1.5,1）分布偏向于较大的值，这有助于模型更好的学习去噪过程
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # 扩展时间维度，使其与动作序列维度匹配
        time_expanded = time[..., None, None]
        
        # 实现了扩散模型的前向过程：计算带噪声的动作序列 x_t
        # 1. time_expanded 是时间步 t 的扩展，范围在 (0.001, 1.0) 之间
        # 2. 当 t 接近 1 时，x_t 主要由噪声组成
        # 3. 当 t 接近 0 时，x_t 主要由原始动作组成
        # 4. 这种线性插值确保了平滑的扩散过程
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        
        # 计算模型需要预测的目标值 u_t
        # 在扩散模型中，我们不是直接预测噪声，而是预测噪声与原始动作的差异
        # 1. noise - actions 表示噪声与原始动作的差异
        # 2. 这个差异值 u_t 作为模型的学习目标
        # 3. 在推理时，模型预测这个差异值，然后通过 x_t - u_t 来恢复原始动作
        # 4. 这种设计使得模型可以更好地学习去噪过程，因为它直接学习噪声与原始动作的关系
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        # 处理模型的前缀部分（视觉和语言输入）
        # prefix_mask    存储每个输入的掩码，用于标记哪些位置是有效的输入（非填充）
        # prefix_ar_mask 存储自回归掩码，用于控制token之间的注意力关系
        # prefix_tokens  存储处理后的token，包含图像和文本的特征表示
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # 处理模型的后缀部分（状态和带噪声的动作）
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        #  将前缀和后缀的掩码在序列维度上拼接
        # prefix_mask : 来自 embed_prefix 函数，包含了图像和文本输入的掩码
        # suffix_mask : 来自 embed_suffix 函数，包含了状态和动作序列的掩码
        # 拼接的目的：
        # - 创建一个完整的输入掩码，覆盖整个序列（包括前缀和后缀部分）
        # - 确保模型知道哪些位置是有效的输入，哪些是填充位置
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        
        # 将前缀和后缀的自回归掩码拼接
        # - prefix_ar_mask : 来自 embed_prefix 函数，控制图像和文本token之间的注意力关系
        # - suffix_ar_mask : 来自 embed_suffix 函数，控制状态和动作token之间的注意力关系
        # 拼接的目的：
        # - 创建一个完整的自回归掩码，覆盖整个序列（包括前缀和后缀部分）
        # - 维护了不同部分token之间的注意力流动规则
        # - axis=0表示在序列维度上拼接，这保持了token的顺序：[图像tokens，文本tokens，状态tokens，动作tokens]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        
        # 生成注意力掩码，控制token之间的注意力关系，确保模型只在有效的token之间计算注意力
        # TODO（tzhang）：两个mask如何生成有效的atten mask？
        attn_mask = make_attn_mask(input_mask, ar_mask)
        
        # 计算每个token的位置编码
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # 通过语言模型处理前缀和后缀token
        # prefix_tokens : 前缀序列，通常包含图像和文本的token表示
        # suffix_tokens : 后缀序列，通常包含状态和动作的token表示
        # attn_mask     : 注意力掩码，用于控制不同token之间的注意力关系
        # positions     : 位置编码，帮助模型理解序列中token的相对位置
        # input_mask    : 输入掩码，用于标记有效输入
        # prefix_out为什么没有被使用？
        # 因为扩散模型的训练过程中，我们的主要目标是预测动作序列的噪声，动作序列的信息都在 suffix_out 中，所以不需要额外使用它。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        
        # 从后缀输出中提取动作相关的部分，并通过投影层得到预测值
        # 这里使用 -self.action_horizon 是因为：
        # 1. suffix_out 包含了动作token的输出
        # 2. 动作token位于序列的最后 action_horizon 个位置
        # 3. 我们只需要这些动作相关的输出来进行噪声预测
        # 4. 通过 action_out_proj 将高维特征投影到动作空间维度
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 计算预测值与目标值之间的均方误差，并在动作维度上取平均
        # v_t 是模型预测的噪声值
        # u_t 是实际的噪声值
        # v_t - u_t 的含义：表示模型预测的噪声与实际噪声之间的误差
        # jnp.square(v_t - u_t)：计算预测误差的平方，这是一个常见的均方误差(MSE)损失的组成部分
        # 这个损失函数的意义：
        # - 它衡量了模型在预测噪声时的准确程度
        # - 损失值越小，表示模型越准确地预测出动作序列中的噪声
        # - 在训练过程中，模型会通过最小化这个损失来学习如何更好地去噪
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
