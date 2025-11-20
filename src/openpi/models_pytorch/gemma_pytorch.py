"""该脚本实现了一个结合PaliGemma视觉语言模型和Gemma专家模型的混合架构
该模型主要用于处理多模态输入（图像和文本），并能够执行特定的动作预测任务。模型架构包含两个主要组件：
1. PaliGemma - 处理视觉和语言输入的视觉语言模型
2. Gemma专家模型 - 专门用于动作预测的语言模型
该实现支持梯度检查点（gradient checkpointing）以减少内存使用，并提供了精度控制功能，允许在bfloat16和float32之间切换以平衡性能和精度。
"""
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM   # 导入Gemma因果语言模型
from transformers import PaliGemmaForConditionalGeneration  # 导入PaliGemma条件生成模型
from transformers.models.auto import CONFIG_MAPPING # 导入模型配置映射
from transformers.models.gemma import modeling_gemma    # 导入Gemma模型相关功能


class PaliGemmaWithExpertModel(nn.Module):
    """
    PaliGemmaWithExpertModel类实现了一个结合PaliGemma视觉语言模型和Gemma专家模型的混合架构。
    该模型能够处理多模态输入（图像和文本），并执行特定的动作预测任务。模型由两个主要组件组成：
    1. PaliGemma - 处理视觉和语言输入的视觉语言模型
    2. Gemma专家模型 - 专门用于动作预测的语言模型
    模型支持梯度检查点以减少内存使用，并提供精度控制功能。
    """
    def __init__(
        self,
        vlm_config,             # 视觉语言模型的配置对象
        action_expert_config,   # 动作专家模型的配置对象
        use_adarms=None,        # 是否使用AdARMS适应性归一化，默认为None，如果为None则设置为[False, False]
        precision: Literal["bfloat16", "float32"] = "bfloat16", # 模型精度，默认为bfloat16
    ):
        if use_adarms is None:
            use_adarms = [False, False]     # 默认不使用AdARMS
        super().__init__()

        # 创建PaliGemma配置
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()       # 创建PaliGemma配置对象
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001   设置词汇表大小
        vlm_config_hf.image_token_index = 257152    # 设置图像令牌索引
        vlm_config_hf.text_config.hidden_size = vlm_config.width    # 设置隐藏层大小
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim    # 设置中间层大小
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads    # 设置注意力头数量
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim        # 设置每个头的维度
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth      # 设置隐藏层数量
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads     # 设置键值头数量
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"   # 设置激活函数
        vlm_config_hf.text_config.torch_dtype = "float32"   # 设置数据类型
        vlm_config_hf.text_config.vocab_size = 257152   # 设置词汇表大小
        vlm_config_hf.text_config.use_adarms = use_adarms[0]    # 设置是否使用AdARMS
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None     # 设置AdARMS条件维度
        vlm_config_hf.vision_config.intermediate_size = 4304    # 设置视觉模型中间层大小
        vlm_config_hf.vision_config.projection_dim = 2048   # 设置投影层维度
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"  # 设置投影器激活函数
        vlm_config_hf.vision_config.torch_dtype = "float32" # 设置视觉模型数据类型

        # 创建Gemma专家模型配置
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,     # 设置头维度
            hidden_size=action_expert_config.width,     # 设置隐藏层大小
            intermediate_size=action_expert_config.mlp_dim,    # 设置中间层大小
            num_attention_heads=action_expert_config.num_heads,   # 设置注意力头数量
            num_hidden_layers=action_expert_config.depth,    # 设置隐藏层数量
            num_key_value_heads=action_expert_config.num_kv_heads,  # 设置键值头数量
            vocab_size=257152,    # 设置词汇表大小
            hidden_activation="gelu_pytorch_tanh",   # 设置隐藏层激活函数
            torch_dtype="float32",   # 设置数据类型
            use_adarms=use_adarms[1],    # 设置是否使用AdARMS
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,   # 设置AdARMS条件维度
        )

        # 初始化模型组件
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)     # 初始化PaliGemma模型
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)    # 初始化Gemma专家模型
        self.gemma_expert.model.embed_tokens = None     # 禁用Gemma专家模型的词嵌入，将使用PaliGemma的嵌入

        # 设置模型精度
        self.to_bfloat16_for_selected_params(precision) # 根据指定精度设置模型参数

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """
        根据指定的精度设置模型参数，某些特定参数保持float32精度以提高稳定性。
        参数:
            precision: 模型精度，可选"bfloat16"或"float32"，默认为"bfloat16"
        返回:
            无返回值，直接修改模型参数
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)   # 将整个模型转换为bfloat16精度
        elif precision == "float32":
            self.to(dtype=torch.float32)    # 将整个模型转换为float32精度
            return  # 如果是float32精度，不需要特殊处理，直接返回
        else:
            raise ValueError(f"Invalid precision: {precision}")     # 无效精度值，抛出异常

        # 定义需要保持float32精度的参数名称列表
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",  # 视觉模型的图像块嵌入权重
            "vision_tower.vision_model.embeddings.patch_embedding.bias",    # 视觉模型的图像块嵌入偏置
            "vision_tower.vision_model.embeddings.position_embedding.weight",   # 视觉模型的位置嵌入权重
            "input_layernorm",  # 输入层归一化
            "post_attention_layernorm",  # 注意力后的层归一化
            "model.norm",    # 模型归一化
        ]

        # 遍历所有参数，将特定参数转换回float32精度
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32) # 将参数转换为float32精度

    def embed_image(self, image: torch.Tensor):
        """获取图像的特征嵌入。
        参数:
            image: 输入图像张量
        返回:
            图像特征嵌入
        """
        return self.paligemma.model.get_image_features(image)   # 使用PaliGemma模型获取图像特征

    def embed_language_tokens(self, tokens: torch.Tensor):
        """获取语言令牌的嵌入。
        参数:
            tokens: 输入令牌张量
        返回:
            语言令牌嵌入
        """
        return self.paligemma.language_model.embed_tokens(tokens)   # 使用PaliGemma语言模型获取令牌嵌入

    def forward(
        self,
        attention_mask: torch.Tensor | None = None, # 注意力掩码
        position_ids: torch.LongTensor | None = None,   # 位置ID
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,  # 过去的键值对
        inputs_embeds: list[torch.FloatTensor] | None = None,   # 输入嵌入
        use_cache: bool | None = None,  # 是否使用缓存
        adarms_cond: list[torch.Tensor] | None = None,  # AdARMS条件
    ):
        """模型前向传播函数。
        根据输入嵌入的情况，有三种不同的处理路径：
        1. 只有PaliGemma输入（inputs_embeds[1]为None）
        2. 只有Gemma专家输入（inputs_embeds[0]为None）
        3. 两种输入都存在 - 执行联合处理
        参数:
            attention_mask: 注意力掩码张量，指定哪些位置应被关注
            position_ids: 位置ID张量，用于位置编码
            past_key_values: 过去的键值对，用于缓存先前计算结果
            inputs_embeds: 输入嵌入列表，包含PaliGemma和Gemma专家的输入嵌入
            use_cache: 是否使用缓存
            adarms_cond: AdARMS条件列表，用于条件归一化

        返回:
            包含PaliGemma和Gemma专家输出的列表，以及PaliGemma的过去键值对
        """
        if adarms_cond is None:
            adarms_cond = [None, None]  # 默认AdARMS条件为None
            
        # 情况1：只有PaliGemma输入（视觉语言模型）
        if inputs_embeds[1] is None:
            # 使用PaliGemma语言模型处理输入
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0], # PaliGemma输入嵌入
                attention_mask=attention_mask,  # 注意力掩码
                position_ids=position_ids,  # 位置ID
                past_key_values=past_key_values,    # 过去的键值对
                use_cache=use_cache,    # 是否使用缓存
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,    # AdARMS条件
            )
            prefix_past_key_values = prefix_output.past_key_values      # 获取过去的键值对
            prefix_output = prefix_output.last_hidden_state     # 获取最后的隐藏状态
            suffix_output = None    # Gemma专家输出为None
        # 情况2：只有Gemma专家输入
        elif inputs_embeds[0] is None:
            # 使用Gemma专家模型处理输入
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1], # Gemma专家输入嵌入
                attention_mask=attention_mask,  # 注意力掩码
                position_ids=position_ids,  # 位置ID
                past_key_values=past_key_values,    # 过去的键值对
                use_cache=use_cache,    # 是否使用缓存
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,    # AdARMS条件
            )
            suffix_output = suffix_output.last_hidden_state # 获取最后的隐藏状态
            prefix_output = None    # PaliGemma输出为Non
            prefix_past_key_values = None   # 过去的键值对为None
        # 情况3：两种输入都存在 - 执行联合处理
        else:
            # 设置模型列表和层数
            models = [self.paligemma.language_model, self.gemma_expert.model]    # 模型列表
            num_layers = self.paligemma.config.text_config.num_hidden_layers    # 层数

            # Check if gradient checkpointing is enabled for any of the models
            # 检查是否启用梯度检查点
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")  # Gemma专家模型是否有梯度检查点属性
                and self.gemma_expert.model.gradient_checkpointing  # Gemma专家模型是否启用梯度检查点
                and self.training   # 是否处于训练模式
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            # 如果处于训练模式且Gemma专家模型支持梯度检查点，则强制启用梯度检查点
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")    # 打印强制启用梯度检查点的信息
                    self.gemma_expert.model.gradient_checkpointing = True   # 强制启用梯度检查点
                use_gradient_checkpointing = True   # 设置使用梯度检查点

            # Debug gradient checkpointing status
            # 调试梯度检查点状态
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")   # 打印Gemma专家模型梯度检查点状态
                print(f"Model training mode: {self.training}")  # 打印模型训练模式
                print(  # 打印Gemma专家模型是否有梯度检查点属性
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(  # 打印Gemma专家模型梯度检查点值
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True   # 设置已打印调试信息

            # Define the complete layer computation function for gradient checkpointing
            # 定义完整层计算函数，用于梯度检查点
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                """计算特定层的完整前向传播，包括自注意力和前馈网络。
                参数:
                    layer_idx: 层索引
                    inputs_embeds: 输入嵌入列表
                    attention_mask: 注意力掩码
                    position_ids: 位置ID
                    adarms_cond: AdaRMS条件列表
                返回:
                    层输出嵌入列表
                """
                models = [self.paligemma.language_model, self.gemma_expert.model]   # 模型列表

                # 初始化状态列表
                query_states = []   # 查询状态列表
                key_states = []     # 键状态列表
                value_states = []   # 值状态列表
                gates = []          # 门控列表
                
                # 对每个模型的输入进行处理
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx] # 获取当前层
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901 输入层归一化
                    gates.append(gate)  # 添加门控

                    # 计算注意力的查询、键、值
                    input_shape = hidden_states.shape[:-1]  # 输入形状
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)     # 隐藏形状
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # 查询投影
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # 键投影
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # 值投影

                    # 添加到状态列表
                    query_states.append(query_state)    # 添加查询状态
                    key_states.append(key_state)    # 添加键状态
                    value_states.append(value_state)    # 添加值状态

                # Concatenate and process attention
                # 连接并处理注意力
                query_states = torch.cat(query_states, dim=2)   # 连接查询状态
                key_states = torch.cat(key_states, dim=2)   # 连接键状态
                value_states = torch.cat(value_states, dim=2)   # 连接值状态

                # 创建虚拟张量用于旋转位置嵌入
                dummy_tensor = torch.zeros(
                    query_states.shape[0],  # 批次大小
                    query_states.shape[2],  # 序列长度
                    query_states.shape[-1], # 隐藏维度
                    device=query_states.device, # 设备
                    dtype=query_states.dtype,   # 数据类型
                )
                # 计算旋转位置嵌入的余弦和正弦值
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                # 应用旋转位置嵌入
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]  # 批次大小
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling     # 缩放因子

                # Attention computation
                # 计算注意力
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,   # 自注意力层
                    query_states,   # 查询状态
                    key_states,     # 键状态
                    value_states,   # 值状态
                    attention_mask, # 注意力掩码
                    scaling,        # 缩放因子
                )
                # Get head_dim from the current layer, not from the model
                # 获取当前层的头维度，而不是从模型获取
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                # 重新塑造注意力输出
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                # 处理层输出
                outputs_embeds = []    # 输出嵌入列表
                start_pos = 0          # 起始位置
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]     # 获取当前层
                    end_pos = start_pos + hidden_states.shape[1]     # 结束位置

                    # 确保数据类型匹配
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)     # 转换数据类型
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])      # 输出投影

                    # first residual
                    # 第一个残差连接
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()  # 保存第一个残差连接后的状态
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])    # 注意力后的层归一化
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    # 如果下一层(mlp)使用bfloat16，则转换为bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    # 前馈网络
                    out_emb = layer.mlp(out_emb)
                    # second residual
                    # 第二个残差连接
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001 门控残差连接
                    outputs_embeds.append(out_emb)  # 添加输出嵌入
                    start_pos = end_pos # 更新起始位置

                return outputs_embeds    # 返回输出嵌入列表

            # Process all layers with gradient checkpointing if enabled
            # 使用梯度检查点处理所有层
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # 使用梯度检查点
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete, # 计算函数
                        layer_idx,  # 层索引
                        inputs_embeds,  # 输入嵌入
                        attention_mask, # 注意力掩码
                        position_ids,   # 位置ID
                        adarms_cond,    # AdaRMS条件
                        use_reentrant=False,    # 不使用重入
                        preserve_rng_state=False,   # 不保留随机数生成器状态
                    )
                else:   # 直接计算
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm 最终归一化
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                """定义最终归一化计算函数，用于梯度检查点。
                计算最终的归一化层。
                参数:
                    inputs_embeds: 输入嵌入列表
                    adarms_cond: AdaRMS条件列表
                返回:
                    归一化后的输出嵌入列表
                """
                outputs_embeds = [] # 输出嵌入列表
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i]) # 应用归一化
                    outputs_embeds.append(out_emb)  # 添加输出嵌入
                return outputs_embeds   # 返回输出嵌入列表

            # Apply gradient checkpointing to final norm if enabled
            # 如果启用梯度检查点，则对最终归一化应用梯度检查点
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # 获取输出
            prefix_output = outputs_embeds[0]   # PaliGemma输出
            suffix_output = outputs_embeds[1]   # Gemma专家输出
            prefix_past_key_values = None   # 过去的键值对为None

        # 返回输出和过去的键值对
        return [prefix_output, suffix_output], prefix_past_key_values
