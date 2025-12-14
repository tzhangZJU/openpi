"""
生成OpenPI模型架构可视化图表
Generate OpenPI Model Architecture Visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
COLORS = {
    'vision': '#3498db',      # 蓝色 - 视觉
    'language': '#2ecc71',    # 绿色 - 语言
    'state': '#f39c12',       # 橙色 - 状态
    'action': '#e74c3c',      # 红色 - 动作
    'attention': '#9b59b6',   # 紫色 - 注意力
    'flow': '#1abc9c',        # 青色 - Flow
    'background': '#ecf0f1'   # 灰色 - 背景
}


def create_pi0_architecture():
    """创建Pi0模型架构图"""
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 标题
    ax.text(5, 19.5, 'Pi0 Model Architecture',
            ha='center', va='top', fontsize=20, fontweight='bold')
    ax.text(5, 19, 'Pi0 模型架构',
            ha='center', va='top', fontsize=16, color='gray')

    y = 18

    # ==================== 输入层 ====================
    # 视觉输入
    vision_box = FancyBboxPatch((0.5, y-0.8), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['vision'], alpha=0.3,
                                edgecolor=COLORS['vision'], linewidth=2)
    ax.add_patch(vision_box)
    ax.text(1.75, y-0.5, 'Images\n3 views × [224,224,3]',
            ha='center', va='center', fontsize=10)

    # 语言输入
    lang_box = FancyBboxPatch((3.5, y-0.8), 2.5, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['language'], alpha=0.3,
                              edgecolor=COLORS['language'], linewidth=2)
    ax.add_patch(lang_box)
    ax.text(4.75, y-0.5, 'Language Prompt\n[batch, seq_len]',
            ha='center', va='center', fontsize=10)

    # 状态输入
    state_box = FancyBboxPatch((6.5, y-0.8), 1.5, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['state'], alpha=0.3,
                               edgecolor=COLORS['state'], linewidth=2)
    ax.add_patch(state_box)
    ax.text(7.25, y-0.5, 'State\n[batch, s]',
            ha='center', va='center', fontsize=10)

    # 噪声动作
    noise_box = FancyBboxPatch((8.5, y-0.8), 1, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['action'], alpha=0.3,
                               edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(noise_box)
    ax.text(9, y-0.5, 'x_t\n[B,H,D]',
            ha='center', va='center', fontsize=10)

    y -= 1.5

    # ==================== 编码层 ====================
    # SigLIP
    siglip_box = FancyBboxPatch((0.5, y-1), 2.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['vision'], alpha=0.5,
                                edgecolor=COLORS['vision'], linewidth=2)
    ax.add_patch(siglip_box)
    ax.text(1.75, y-0.6, 'SigLIP Encoder\nSo400m/14',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.75, y-0.9, '↓\nVision Tokens\n[B, 768, 1152]',
            ha='center', va='top', fontsize=8)

    # Tokenizer
    tok_box = FancyBboxPatch((3.5, y-1), 2.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['language'], alpha=0.5,
                             edgecolor=COLORS['language'], linewidth=2)
    ax.add_patch(tok_box)
    ax.text(4.75, y-0.6, 'Text Embeddings\nPaliGemma',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.75, y-0.9, '↓\nText Tokens\n[B, L, 1152]',
            ha='center', va='top', fontsize=8)

    # State Projection
    state_proj_box = FancyBboxPatch((6.5, y-1), 1.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['state'], alpha=0.5,
                                    edgecolor=COLORS['state'], linewidth=2)
    ax.add_patch(state_proj_box)
    ax.text(7.25, y-0.6, 'Linear\nProjection',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, y-0.9, '↓\n[B, 1, 1152]',
            ha='center', va='top', fontsize=8)

    # Action + Time MLP
    action_mlp_box = FancyBboxPatch((8.5, y-1.2), 1, 1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['action'], alpha=0.5,
                                    edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(action_mlp_box)
    ax.text(9, y-0.5, 'Action\n+\nTime\nMLP',
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(9, y-1, '↓\n[B,H,1152]',
            ha='center', va='top', fontsize=7)

    y -= 2

    # ==================== Token拼接 ====================
    # 前缀
    prefix_box = FancyBboxPatch((0.5, y-0.8), 5.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['attention'], alpha=0.2,
                                edgecolor=COLORS['attention'], linewidth=2,
                                linestyle='--')
    ax.add_patch(prefix_box)
    ax.text(3.25, y-0.5, 'Prefix Tokens: [Vision | Language]',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.25, y-0.75, 'Attention: Bidirectional (全连接)',
            ha='center', va='center', fontsize=8, style='italic')

    # 后缀
    suffix_box = FancyBboxPatch((6.5, y-0.8), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['action'], alpha=0.2,
                                edgecolor=COLORS['action'], linewidth=2,
                                linestyle='--')
    ax.add_patch(suffix_box)
    ax.text(8, y-0.5, 'Suffix: [State | Actions]',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8, y-0.75, 'Attention: Causal (因果)',
            ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # ==================== Transformer层 ====================
    # PaliGemma
    pali_box = FancyBboxPatch((0.5, y-2.5), 4, 2.3,
                              boxstyle="round,pad=0.15",
                              facecolor='#3498db', alpha=0.15,
                              edgecolor='#3498db', linewidth=2.5)
    ax.add_patch(pali_box)
    ax.text(2.5, y-0.3, 'PaliGemma (3B)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # PaliGemma层
    for i in range(3):
        layer_y = y - 0.7 - i * 0.6
        layer_box = Rectangle((0.8, layer_y-0.4), 3.4, 0.35,
                              facecolor='white', edgecolor='#2980b9', linewidth=1.5)
        ax.add_patch(layer_box)
        ax.text(2.5, layer_y-0.225, f'Layer {i+1}: MHA + FFN',
                ha='center', va='center', fontsize=8)

    ax.text(2.5, y-2.4, '... (24 layers total)',
            ha='center', va='center', fontsize=7, style='italic')

    # Action Expert
    expert_box = FancyBboxPatch((5, y-2.5), 4, 2.3,
                                boxstyle="round,pad=0.15",
                                facecolor='#e74c3c', alpha=0.15,
                                edgecolor='#e74c3c', linewidth=2.5)
    ax.add_patch(expert_box)
    ax.text(7, y-0.3, 'Action Expert (2B)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Action Expert层
    for i in range(3):
        layer_y = y - 0.7 - i * 0.6
        layer_box = Rectangle((5.3, layer_y-0.4), 3.4, 0.35,
                              facecolor='white', edgecolor='#c0392b', linewidth=1.5)
        ax.add_patch(layer_box)
        ax.text(7, layer_y-0.225, f'Layer {i+1}: MHA + FFN',
                ha='center', va='center', fontsize=8)

    ax.text(7, y-2.4, '... (18 layers total)',
            ha='center', va='center', fontsize=7, style='italic')

    y -= 3

    # ==================== 输出层 ====================
    # 提取动作特征
    extract_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLORS['action'], alpha=0.3,
                                 edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(extract_box)
    ax.text(7, y-0.5, 'Extract Action Features\n[B, H, 1152]',
            ha='center', va='center', fontsize=9)

    y -= 1.2

    # 动作投影
    proj_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['action'], alpha=0.5,
                              edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(proj_box)
    ax.text(7, y-0.5, 'Action Projection\nLinear(1152 → action_dim)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    y -= 1.2

    # 速度场
    velocity_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=COLORS['flow'], alpha=0.5,
                                  edgecolor=COLORS['flow'], linewidth=2)
    ax.add_patch(velocity_box)
    ax.text(7, y-0.5, 'Velocity Field v_t\n[B, H, action_dim]',
            ha='center', va='center', fontsize=10, fontweight='bold')

    y -= 1.2

    # 损失
    loss_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#95a5a6', alpha=0.5,
                              edgecolor='#7f8c8d', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, y-0.5, 'Loss = MSE(v_t, u_t)\nu_t = noise - actions',
            ha='center', va='center', fontsize=9)

    # 添加连接箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='#34495e')

    # 输入到编码器
    ax.annotate('', xy=(1.75, 16.5), xytext=(1.75, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, 16.5), xytext=(4.75, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 16.5), xytext=(7.25, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 16.5), xytext=(9, 17.2), arrowprops=arrow_props)

    # 编码器到tokens
    ax.annotate('', xy=(1.75, 14.7), xytext=(1.75, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, 14.7), xytext=(4.75, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 14.7), xytext=(7.25, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 14.5), xytext=(9, 15.5), arrowprops=arrow_props)

    # Tokens到Transformer
    ax.annotate('', xy=(2.5, 11.5), xytext=(2.5, 13.2), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 11.5), xytext=(7, 13.2), arrowprops=arrow_props)

    # Transformer到输出
    ax.annotate('', xy=(7, 7.7), xytext=(7, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 6.5), xytext=(7, 7.1), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 5.3), xytext=(7, 5.9), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 4.1), xytext=(7, 4.7), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig('pi0_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Pi0架构图已保存: pi0_architecture.png")
    plt.close()


def create_flow_matching_diagram():
    """创建Flow Matching训练和推理流程图"""
    fig = plt.figure(figsize=(16, 10))

    # 训练流程
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.text(5, 11.5, 'Flow Matching Training',
             ha='center', va='top', fontsize=16, fontweight='bold')
    ax1.text(5, 11, '流匹配训练过程',
             ha='center', va='top', fontsize=12, color='gray')

    y = 10

    # 1. 数据和噪声
    data_box = FancyBboxPatch((1, y-0.6), 3.5, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#2ecc71', alpha=0.3,
                              edgecolor='#27ae60', linewidth=2)
    ax1.add_patch(data_box)
    ax1.text(2.75, y-0.35, 'Data: actions ~ p_data',
             ha='center', va='center', fontsize=10)

    noise_box = FancyBboxPatch((5.5, y-0.6), 3.5, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#3498db', alpha=0.3,
                               edgecolor='#2980b9', linewidth=2)
    ax1.add_patch(noise_box)
    ax1.text(7.25, y-0.35, 'Noise: ε ~ N(0, I)',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 2. 时间采样
    time_box = FancyBboxPatch((2.5, y-0.6), 5, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#f39c12', alpha=0.3,
                              edgecolor='#e67e22', linewidth=2)
    ax1.add_patch(time_box)
    ax1.text(5, y-0.35, 'Sample t ~ Beta(1.5, 1) * 0.999 + 0.001',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 3. 插值
    interp_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#9b59b6', alpha=0.3,
                                edgecolor='#8e44ad', linewidth=2)
    ax1.add_patch(interp_box)
    ax1.text(5, y-0.55, 'x_t = t · ε + (1-t) · actions',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.75, '(Linear interpolation / 线性插值)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 4. 目标速度
    target_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#1abc9c', alpha=0.3,
                                edgecolor='#16a085', linewidth=2)
    ax1.add_patch(target_box)
    ax1.text(5, y-0.55, 'u_t = ε - actions',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.75, '(Target velocity / 目标速度场)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 5. 模型预测
    model_box = FancyBboxPatch((1.5, y-1), 7, 0.9,
                               boxstyle="round,pad=0.15",
                               facecolor='#e74c3c', alpha=0.3,
                               edgecolor='#c0392b', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(5, y-0.4, 'Model Prediction',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.65, 'v_t = Pi0(obs, x_t, t)',
             ha='center', va='center', fontsize=10)
    ax1.text(5, y-0.85, '(Predicted velocity / 预测速度场)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 6. 损失
    loss_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor='#95a5a6', alpha=0.5,
                              edgecolor='#7f8c8d', linewidth=3)
    ax1.add_patch(loss_box)
    ax1.text(5, y-0.55, 'Loss = MSE(v_t, u_t)',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, y-0.75, '= ||v_t - u_t||²',
             ha='center', va='center', fontsize=9)

    # 添加箭头
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#34495e')
    for i in range(5):
        start_y = 10 - i * 1.2 - 0.6
        end_y = start_y - 0.6
        ax1.annotate('', xy=(5, end_y), xytext=(5, start_y), arrowprops=arrow_props)

    # 推理流程
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.text(5, 11.5, 'Flow Matching Inference',
             ha='center', va='top', fontsize=16, fontweight='bold')
    ax2.text(5, 11, '流匹配推理过程 (ODE求解)',
             ha='center', va='top', fontsize=12, color='gray')

    y = 10

    # 1. 初始化
    init_box = FancyBboxPatch((2, y-0.6), 6, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#3498db', alpha=0.3,
                              edgecolor='#2980b9', linewidth=2)
    ax2.add_patch(init_box)
    ax2.text(5, y-0.35, 'x_1 = ε ~ N(0, I),  t = 1.0',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 2. ODE步骤
    ode_box = FancyBboxPatch((1, y-2.5), 8, 2.3,
                             boxstyle="round,pad=0.15",
                             facecolor='#9b59b6', alpha=0.15,
                             edgecolor='#8e44ad', linewidth=2.5)
    ax2.add_patch(ode_box)
    ax2.text(5, y-0.3, 'ODE Solving Loop (循环求解)',
             ha='center', va='center', fontsize=11, fontweight='bold')

    step_y = y - 0.7
    for i in range(3):
        step_box = Rectangle((1.5, step_y - i*0.55 - 0.4), 7, 0.4,
                             facecolor='white', edgecolor='#8e44ad', linewidth=1.5)
        ax2.add_patch(step_box)

        if i < 2:
            ax2.text(5, step_y - i*0.55 - 0.2,
                    f'Step {i+1}: v_t = Model(x_t, t); x_t -= dt·v_t; t -= dt',
                    ha='center', va='center', fontsize=8)
        else:
            ax2.text(5, step_y - i*0.55 - 0.2,
                    '...',
                    ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.text(5, y-2.4, f'Repeat num_steps times (default: 10)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 3

    # 3. 终止条件
    cond_box = FancyBboxPatch((2, y-0.6), 6, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#f39c12', alpha=0.3,
                              edgecolor='#e67e22', linewidth=2)
    ax2.add_patch(cond_box)
    ax2.text(5, y-0.35, 'Until t ≈ 0',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 4. 输出
    output_box = FancyBboxPatch((2, y-0.8), 6, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#2ecc71', alpha=0.5,
                                edgecolor='#27ae60', linewidth=3)
    ax2.add_patch(output_box)
    ax2.text(5, y-0.55, 'Output: x_0 ≈ actions',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(5, y-0.75, '(Clean actions / 干净的动作序列)',
             ha='center', va='center', fontsize=8, style='italic')

    # 添加箭头
    ax2.annotate('', xy=(5, 9.4), xytext=(5, 10), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 5.8), xytext=(5, 8.8), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 4.6), xytext=(5, 5.2), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 3.2), xytext=(5, 4), arrowprops=arrow_props)

    # 添加说明文字
    ax2.text(7.5, 7.3, 'dt = -1/num_steps\n(负数，从1到0)',
             ha='left', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('flow_matching_process.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Flow Matching流程图已保存: flow_matching_process.png")
    plt.close()


def create_attention_mask_diagram():
    """创建注意力掩码可视化图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 示例序列
    seq_labels = ['img0', 'img1', 'img2', 'text0', 'text1', 'state', 'act0', 'act1', 'act2']
    ar_mask = [False, False, False, False, False, True, True, False, False]

    # 计算累积和
    cumsum = np.cumsum(ar_mask)

    # 构造注意力掩码
    n = len(seq_labels)
    attn_mask = cumsum[None, :] <= cumsum[:, None]

    # 绘制累积和
    ax1.set_title('AR Mask Cumulative Sum\nar_mask累积和', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(-0.5, 2.5)

    # 绘制token blocks
    colors_list = [COLORS['vision']]*3 + [COLORS['language']]*2 + \
                  [COLORS['state']] + [COLORS['action']]*3

    for i, (label, val, color) in enumerate(zip(seq_labels, cumsum, colors_list)):
        rect = Rectangle((i-0.4, -0.4), 0.8, 0.8,
                        facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(i, 0, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax1.text(i, 1, str(val), ha='center', va='center', fontsize=14, fontweight='bold')

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Token', 'Cumsum'])
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([])
    ax1.grid(axis='x', alpha=0.3)

    # 绘制注意力掩码矩阵
    ax2.set_title('Attention Mask Matrix\n注意力掩码矩阵', fontsize=14, fontweight='bold')

    # 使用热图
    im = ax2.imshow(attn_mask.astype(float), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 设置标签
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(seq_labels, rotation=45, ha='right')
    ax2.set_yticklabels(seq_labels)

    ax2.set_xlabel('Key (被attend的token)', fontsize=10)
    ax2.set_ylabel('Query (发起attend的token)', fontsize=10)

    # 添加网格
    for i in range(n+1):
        ax2.axhline(i-0.5, color='gray', linewidth=0.5)
        ax2.axvline(i-0.5, color='gray', linewidth=0.5)

    # 添加数值标注
    for i in range(n):
        for j in range(n):
            text = ax2.text(j, i, '✓' if attn_mask[i, j] else '✗',
                          ha='center', va='center',
                          color='darkgreen' if attn_mask[i, j] else 'darkred',
                          fontsize=10, fontweight='bold')

    # 添加图例
    ax2.text(n+0.5, 1, '✓ = Can attend\n可以关注',
             ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax2.text(n+0.5, 3, '✗ = Cannot attend\n不可关注',
             ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

    # 添加区域标注
    # 前缀块
    ax2.add_patch(Rectangle((-0.5, -0.5), 5, 5,
                            fill=False, edgecolor=COLORS['attention'],
                            linewidth=3, linestyle='--'))
    ax2.text(2, -1, 'Prefix Block\n(全连接)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color=COLORS['attention'])

    # 后缀块
    ax2.add_patch(Rectangle((4.5, 4.5), 4, 4,
                            fill=False, edgecolor=COLORS['action'],
                            linewidth=3, linestyle='--'))
    ax2.text(6.5, n, 'Suffix Block\n(因果)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color=COLORS['action'])

    plt.tight_layout()
    plt.savefig('attention_mask_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 注意力掩码可视化已保存: attention_mask_visualization.png")
    plt.close()


def create_adarms_diagram():
    """创建AdaRMS机制图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    ax.text(5, 11.5, 'AdaRMS Mechanism (Pi0.5)',
            ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(5, 11, '自适应RMS归一化机制',
            ha='center', va='top', fontsize=14, color='gray')

    y = 10

    # 输入
    input_box = FancyBboxPatch((1, y-0.6), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['action'], alpha=0.3,
                               edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, y-0.35, 'Input x\n[B, L, D]',
            ha='center', va='center', fontsize=10)

    cond_box = FancyBboxPatch((6, y-0.6), 3, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['flow'], alpha=0.3,
                              edgecolor=COLORS['flow'], linewidth=2)
    ax.add_patch(cond_box)
    ax.text(7.5, y-0.35, 'Condition (time_emb)\n[B, D]',
            ha='center', va='center', fontsize=10)

    y -= 1.5

    # RMS计算
    rms_box = FancyBboxPatch((0.5, y-1), 4, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor='#3498db', alpha=0.3,
                             edgecolor='#2980b9', linewidth=2)
    ax.add_patch(rms_box)
    ax.text(2.5, y-0.3, 'Compute RMS',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.5, y-0.65, 'rms = sqrt(mean(x²) + ε)',
            ha='center', va='center', fontsize=9)
    ax.text(2.5, y-0.9, 'x_norm = x / rms',
            ha='center', va='center', fontsize=9)

    # 条件投影
    proj_box = FancyBboxPatch((5.5, y-1), 4, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor='#1abc9c', alpha=0.3,
                              edgecolor='#16a085', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(7.5, y-0.3, 'Adaptive Scaling',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.5, y-0.55, 'Linear(cond_dim → D)',
            ha='center', va='center', fontsize=9)
    ax.text(7.5, y-0.75, '↓',
            ha='center', va='center', fontsize=12)
    ax.text(7.5, y-0.9, 'α = MLP(time_emb)',
            ha='center', va='center', fontsize=9)

    y -= 1.5

    # 标准缩放
    gamma_box = FancyBboxPatch((1, y-0.6), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#f39c12', alpha=0.3,
                               edgecolor='#e67e22', linewidth=2)
    ax.add_patch(gamma_box)
    ax.text(2.5, y-0.35, 'Learned Scale γ\n[D]',
            ha='center', va='center', fontsize=10)

    alpha_box = FancyBboxPatch((6, y-0.6), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#9b59b6', alpha=0.3,
                               edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(alpha_box)
    ax.text(7.5, y-0.35, 'Adaptive α\n[B, 1, D]',
            ha='center', va='center', fontsize=10)

    y -= 1.5

    # 组合
    combine_box = FancyBboxPatch((1.5, y-1), 7, 0.9,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#e74c3c', alpha=0.3,
                                 edgecolor='#c0392b', linewidth=3)
    ax.add_patch(combine_box)
    ax.text(5, y-0.3, 'Adaptive Normalization',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y-0.65, 'output = x_norm * (γ + α)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    y -= 1.5

    # 输出
    output_box = FancyBboxPatch((2.5, y-0.8), 5, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#2ecc71', alpha=0.5,
                                edgecolor='#27ae60', linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, y-0.55, 'Output: Normalized Features\n[B, L, D]',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # 添加箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='#34495e')

    ax.annotate('', xy=(2.5, 8.5), xytext=(2.5, 9.4), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 8.5), xytext=(7.5, 9.4), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 6.5), xytext=(2.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 6.5), xytext=(7.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.5), xytext=(2.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.5), xytext=(7.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.6), arrowprops=arrow_props)

    # 添加优势说明
    ax.text(5, 1.5, 'Advantages / 优势:',
            ha='center', va='center', fontsize=12, fontweight='bold')

    advantages = [
        '1. Time信息深度融入每一层 (Deep integration of time info)',
        '2. 动态调整归一化强度 (Dynamic normalization strength)',
        '3. 改善梯度流动 (Improved gradient flow)',
        '4. 更好的条件生成 (Better conditional generation)'
    ]

    for i, adv in enumerate(advantages):
        ax.text(5, 1.1 - i*0.3, adv,
                ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('adarms_mechanism.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ AdaRMS机制图已保存: adarms_mechanism.png")
    plt.close()


def create_model_comparison():
    """创建三种模型对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    models = ['Pi0', 'Pi0-Fast', 'Pi0.5']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (ax, model, color) in enumerate(zip(axes, models, colors)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # 标题
        ax.text(5, 9.5, model,
                ha='center', va='top', fontsize=16, fontweight='bold', color=color)

        y = 8.5

        if model == 'Pi0':
            # Pi0特点
            features = [
                ('动作生成', 'Diffusion Model\n扩散模型'),
                ('时间嵌入', 'MLP Mixing\nMLP混合'),
                ('状态处理', 'Separate Token\n独立token'),
                ('归一化', 'Standard RMSNorm\n标准RMS归一化'),
                ('推理速度', 'Slow (10 steps)\n慢 (10步ODE)')
            ]
        elif model == 'Pi0-Fast':
            features = [
                ('动作生成', 'Autoregressive\n自回归生成'),
                ('Token化', 'VQ Codebook\n向量量化码本'),
                ('状态处理', 'Fused in Prefix\n融入前缀'),
                ('训练', 'Cross Entropy\n交叉熵损失'),
                ('推理速度', 'Fast (1 step)\n快 (单步)')
            ]
        else:  # Pi0.5
            features = [
                ('动作生成', 'Diffusion Model\n扩散模型'),
                ('时间嵌入', 'AdaRMS Conditioning\nAdaRMS条件化'),
                ('状态处理', 'No Separate Token\n无独立token'),
                ('归一化', 'AdaRMS\n自适应RMS'),
                ('推理速度', 'Slow (10 steps)\n慢 (10步ODE)')
            ]

        for i, (label, value) in enumerate(features):
            feature_box = FancyBboxPatch((1, y - i*1.4 - 1), 8, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=2)
            ax.add_patch(feature_box)

            ax.text(2, y - i*1.4 - 0.4, label,
                   ha='left', va='center', fontsize=10, fontweight='bold')
            ax.text(5, y - i*1.4 - 0.65, value,
                   ha='center', va='center', fontsize=9)

    plt.suptitle('OpenPI Model Variants Comparison\nOpenPI模型变体对比',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 模型对比图已保存: model_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("\n=== 开始生成OpenPI架构可视化图表 ===\n")

    create_pi0_architecture()
    create_flow_matching_diagram()
    create_attention_mask_diagram()
    create_adarms_diagram()
    create_model_comparison()

    print("\n=== 所有图表生成完成! ===")
    print("\n生成的文件:")
    print("1. pi0_architecture.png - Pi0模型完整架构")
    print("2. flow_matching_process.png - Flow Matching训练和推理流程")
    print("3. attention_mask_visualization.png - 注意力掩码可视化")
    print("4. adarms_mechanism.png - AdaRMS机制详解")
    print("5. model_comparison.png - 三种模型变体对比")
    print("\n请查看当前目录下的PNG文件。\n")
