"""
Generate OpenPI Model Architecture Visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Remove Chinese font configuration - use default fonts
plt.rcParams['axes.unicode_minus'] = False

# Color configuration
COLORS = {
    'vision': '#3498db',      # Blue - Vision
    'language': '#2ecc71',    # Green - Language
    'state': '#f39c12',       # Orange - State
    'action': '#e74c3c',      # Red - Action
    'attention': '#9b59b6',   # Purple - Attention
    'flow': '#1abc9c',        # Cyan - Flow
    'background': '#ecf0f1'   # Gray - Background
}


def create_pi0_architecture():
    """Create Pi0 model architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Title
    ax.text(5, 19.5, 'Pi0 Model Architecture',
            ha='center', va='top', fontsize=20, fontweight='bold')

    y = 18

    # ==================== Input Layer ====================
    # Vision input
    vision_box = FancyBboxPatch((0.5, y-0.8), 2.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['vision'], alpha=0.3,
                                edgecolor=COLORS['vision'], linewidth=2)
    ax.add_patch(vision_box)
    ax.text(1.75, y-0.5, 'Images\n3 views x [224,224,3]',
            ha='center', va='center', fontsize=10)

    # Language input
    lang_box = FancyBboxPatch((3.5, y-0.8), 2.5, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['language'], alpha=0.3,
                              edgecolor=COLORS['language'], linewidth=2)
    ax.add_patch(lang_box)
    ax.text(4.75, y-0.5, 'Language Prompt\n[batch, seq_len]',
            ha='center', va='center', fontsize=10)

    # State input
    state_box = FancyBboxPatch((6.5, y-0.8), 1.5, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['state'], alpha=0.3,
                               edgecolor=COLORS['state'], linewidth=2)
    ax.add_patch(state_box)
    ax.text(7.25, y-0.5, 'State\n[batch, s]',
            ha='center', va='center', fontsize=10)

    # Noise action
    noise_box = FancyBboxPatch((8.5, y-0.8), 1, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=COLORS['action'], alpha=0.3,
                               edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(noise_box)
    ax.text(9, y-0.5, 'x_t\n[B,H,D]',
            ha='center', va='center', fontsize=10)

    y -= 1.5

    # ==================== Encoding Layer ====================
    # SigLIP
    siglip_box = FancyBboxPatch((0.5, y-1), 2.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['vision'], alpha=0.5,
                                edgecolor=COLORS['vision'], linewidth=2)
    ax.add_patch(siglip_box)
    ax.text(1.75, y-0.6, 'SigLIP Encoder\nSo400m/14',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.75, y-0.9, 'Vision Tokens\n[B, 768, 1152]',
            ha='center', va='top', fontsize=8)

    # Tokenizer
    tok_box = FancyBboxPatch((3.5, y-1), 2.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['language'], alpha=0.5,
                             edgecolor=COLORS['language'], linewidth=2)
    ax.add_patch(tok_box)
    ax.text(4.75, y-0.6, 'Text Embeddings\nPaliGemma',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.75, y-0.9, 'Text Tokens\n[B, L, 1152]',
            ha='center', va='top', fontsize=8)

    # State Projection
    state_proj_box = FancyBboxPatch((6.5, y-1), 1.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['state'], alpha=0.5,
                                    edgecolor=COLORS['state'], linewidth=2)
    ax.add_patch(state_proj_box)
    ax.text(7.25, y-0.6, 'Linear\nProjection',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, y-0.9, '[B, 1, 1152]',
            ha='center', va='top', fontsize=8)

    # Action + Time MLP
    action_mlp_box = FancyBboxPatch((8.5, y-1.2), 1, 1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=COLORS['action'], alpha=0.5,
                                    edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(action_mlp_box)
    ax.text(9, y-0.5, 'Action\n+\nTime\nMLP',
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(9, y-1, '[B,H,1152]',
            ha='center', va='top', fontsize=7)

    y -= 2

    # ==================== Token Concatenation ====================
    # Prefix
    prefix_box = FancyBboxPatch((0.5, y-0.8), 5.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['attention'], alpha=0.2,
                                edgecolor=COLORS['attention'], linewidth=2,
                                linestyle='--')
    ax.add_patch(prefix_box)
    ax.text(3.25, y-0.5, 'Prefix Tokens: [Vision | Language]',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.25, y-0.75, 'Attention: Bidirectional (fully connected)',
            ha='center', va='center', fontsize=8, style='italic')

    # Suffix
    suffix_box = FancyBboxPatch((6.5, y-0.8), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['action'], alpha=0.2,
                                edgecolor=COLORS['action'], linewidth=2,
                                linestyle='--')
    ax.add_patch(suffix_box)
    ax.text(8, y-0.5, 'Suffix: [State | Actions]',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8, y-0.75, 'Attention: Causal',
            ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # ==================== Transformer Layers ====================
    # PaliGemma
    pali_box = FancyBboxPatch((0.5, y-2.5), 4, 2.3,
                              boxstyle="round,pad=0.15",
                              facecolor='#3498db', alpha=0.15,
                              edgecolor='#3498db', linewidth=2.5)
    ax.add_patch(pali_box)
    ax.text(2.5, y-0.3, 'PaliGemma (3B)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # PaliGemma layers
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

    # Action Expert layers
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

    # ==================== Output Layer ====================
    # Extract action features
    extract_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLORS['action'], alpha=0.3,
                                 edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(extract_box)
    ax.text(7, y-0.5, 'Extract Action Features\n[B, H, 1152]',
            ha='center', va='center', fontsize=9)

    y -= 1.2

    # Action projection
    proj_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['action'], alpha=0.5,
                              edgecolor=COLORS['action'], linewidth=2)
    ax.add_patch(proj_box)
    ax.text(7, y-0.5, 'Action Projection\nLinear(1152 -> action_dim)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    y -= 1.2

    # Velocity field
    velocity_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=COLORS['flow'], alpha=0.5,
                                  edgecolor=COLORS['flow'], linewidth=2)
    ax.add_patch(velocity_box)
    ax.text(7, y-0.5, 'Velocity Field v_t\n[B, H, action_dim]',
            ha='center', va='center', fontsize=10, fontweight='bold')

    y -= 1.2

    # Loss
    loss_box = FancyBboxPatch((5.5, y-0.8), 3, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#95a5a6', alpha=0.5,
                              edgecolor='#7f8c8d', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, y-0.5, 'Loss = MSE(v_t, u_t)\nu_t = noise - actions',
            ha='center', va='center', fontsize=9)

    # Add connecting arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#34495e')

    # Input to encoder
    ax.annotate('', xy=(1.75, 16.5), xytext=(1.75, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, 16.5), xytext=(4.75, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 16.5), xytext=(7.25, 17.2), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 16.5), xytext=(9, 17.2), arrowprops=arrow_props)

    # Encoder to tokens
    ax.annotate('', xy=(1.75, 14.7), xytext=(1.75, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, 14.7), xytext=(4.75, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 14.7), xytext=(7.25, 15.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 14.5), xytext=(9, 15.5), arrowprops=arrow_props)

    # Tokens to Transformer
    ax.annotate('', xy=(2.5, 11.5), xytext=(2.5, 13.2), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 11.5), xytext=(7, 13.2), arrowprops=arrow_props)

    # Transformer to output
    ax.annotate('', xy=(7, 7.7), xytext=(7, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 6.5), xytext=(7, 7.1), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 5.3), xytext=(7, 5.9), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 4.1), xytext=(7, 4.7), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig('pi0_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Pi0 architecture saved: pi0_architecture.png")
    plt.close()


def create_flow_matching_diagram():
    """Create Flow Matching training and inference diagrams"""
    fig = plt.figure(figsize=(16, 10))

    # Training flow
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.text(5, 11.5, 'Flow Matching Training',
             ha='center', va='top', fontsize=16, fontweight='bold')

    y = 10

    # 1. Data and noise
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
    ax1.text(7.25, y-0.35, 'Noise: epsilon ~ N(0, I)',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 2. Time sampling
    time_box = FancyBboxPatch((2.5, y-0.6), 5, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#f39c12', alpha=0.3,
                              edgecolor='#e67e22', linewidth=2)
    ax1.add_patch(time_box)
    ax1.text(5, y-0.35, 'Sample t ~ Beta(1.5, 1) * 0.999 + 0.001',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 3. Interpolation
    interp_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#9b59b6', alpha=0.3,
                                edgecolor='#8e44ad', linewidth=2)
    ax1.add_patch(interp_box)
    ax1.text(5, y-0.55, 'x_t = t * epsilon + (1-t) * actions',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.75, '(Linear interpolation)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 4. Target velocity
    target_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#1abc9c', alpha=0.3,
                                edgecolor='#16a085', linewidth=2)
    ax1.add_patch(target_box)
    ax1.text(5, y-0.55, 'u_t = epsilon - actions',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.75, '(Target velocity field)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 5. Model prediction
    model_box = FancyBboxPatch((1.5, y-1), 7, 0.9,
                               boxstyle="round,pad=0.15",
                               facecolor='#e74c3c', alpha=0.3,
                               edgecolor='#c0392b', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(5, y-0.4, 'Model Prediction',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, y-0.65, 'v_t = Pi0(obs, x_t, t)',
             ha='center', va='center', fontsize=10)
    ax1.text(5, y-0.85, '(Predicted velocity field)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 1.5

    # 6. Loss
    loss_box = FancyBboxPatch((1.5, y-0.8), 7, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor='#95a5a6', alpha=0.5,
                              edgecolor='#7f8c8d', linewidth=3)
    ax1.add_patch(loss_box)
    ax1.text(5, y-0.55, 'Loss = MSE(v_t, u_t)',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, y-0.75, '= ||v_t - u_t||^2',
             ha='center', va='center', fontsize=9)

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#34495e')
    for i in range(5):
        start_y = 10 - i * 1.2 - 0.6
        end_y = start_y - 0.6
        ax1.annotate('', xy=(5, end_y), xytext=(5, start_y), arrowprops=arrow_props)

    # Inference flow
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.text(5, 11.5, 'Flow Matching Inference',
             ha='center', va='top', fontsize=16, fontweight='bold')
    ax2.text(5, 11, '(ODE Solving)',
             ha='center', va='top', fontsize=12, color='gray')

    y = 10

    # 1. Initialization
    init_box = FancyBboxPatch((2, y-0.6), 6, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#3498db', alpha=0.3,
                              edgecolor='#2980b9', linewidth=2)
    ax2.add_patch(init_box)
    ax2.text(5, y-0.35, 'x_1 = epsilon ~ N(0, I),  t = 1.0',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 2. ODE steps
    ode_box = FancyBboxPatch((1, y-2.5), 8, 2.3,
                             boxstyle="round,pad=0.15",
                             facecolor='#9b59b6', alpha=0.15,
                             edgecolor='#8e44ad', linewidth=2.5)
    ax2.add_patch(ode_box)
    ax2.text(5, y-0.3, 'ODE Solving Loop',
             ha='center', va='center', fontsize=11, fontweight='bold')

    step_y = y - 0.7
    for i in range(3):
        step_box = Rectangle((1.5, step_y - i*0.55 - 0.4), 7, 0.4,
                             facecolor='white', edgecolor='#8e44ad', linewidth=1.5)
        ax2.add_patch(step_box)

        if i < 2:
            ax2.text(5, step_y - i*0.55 - 0.2,
                    f'Step {i+1}: v_t = Model(x_t, t); x_t -= dt*v_t; t -= dt',
                    ha='center', va='center', fontsize=8)
        else:
            ax2.text(5, step_y - i*0.55 - 0.2,
                    '...',
                    ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.text(5, y-2.4, f'Repeat num_steps times (default: 10)',
             ha='center', va='center', fontsize=8, style='italic')

    y -= 3

    # 3. Termination condition
    cond_box = FancyBboxPatch((2, y-0.6), 6, 0.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#f39c12', alpha=0.3,
                              edgecolor='#e67e22', linewidth=2)
    ax2.add_patch(cond_box)
    ax2.text(5, y-0.35, 'Until t ~= 0',
             ha='center', va='center', fontsize=10)

    y -= 1.2

    # 4. Output
    output_box = FancyBboxPatch((2, y-0.8), 6, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#2ecc71', alpha=0.5,
                                edgecolor='#27ae60', linewidth=3)
    ax2.add_patch(output_box)
    ax2.text(5, y-0.55, 'Output: x_0 ~= actions',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(5, y-0.75, '(Clean action sequence)',
             ha='center', va='center', fontsize=8, style='italic')

    # Add arrows
    ax2.annotate('', xy=(5, 9.4), xytext=(5, 10), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 5.8), xytext=(5, 8.8), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 4.6), xytext=(5, 5.2), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 3.2), xytext=(5, 4), arrowprops=arrow_props)

    # Add notes
    ax2.text(7.5, 7.3, 'dt = -1/num_steps\n(negative: 1 to 0)',
             ha='left', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('flow_matching_process.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Flow Matching process saved: flow_matching_process.png")
    plt.close()


def create_attention_mask_diagram():
    """Create attention mask visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Example sequence
    seq_labels = ['img0', 'img1', 'img2', 'text0', 'text1', 'state', 'act0', 'act1', 'act2']
    ar_mask = [False, False, False, False, False, True, True, False, False]

    # Compute cumulative sum
    cumsum = np.cumsum(ar_mask)

    # Build attention mask
    n = len(seq_labels)
    attn_mask = cumsum[None, :] <= cumsum[:, None]

    # Plot cumulative sum
    ax1.set_title('AR Mask Cumulative Sum', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(-0.5, 2.5)

    # Draw token blocks
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

    # Plot attention mask matrix
    ax2.set_title('Attention Mask Matrix', fontsize=14, fontweight='bold')

    # Use heatmap
    im = ax2.imshow(attn_mask.astype(float), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set labels
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(seq_labels, rotation=45, ha='right')
    ax2.set_yticklabels(seq_labels)

    ax2.set_xlabel('Key (attended token)', fontsize=10)
    ax2.set_ylabel('Query (attending token)', fontsize=10)

    # Add grid
    for i in range(n+1):
        ax2.axhline(i-0.5, color='gray', linewidth=0.5)
        ax2.axvline(i-0.5, color='gray', linewidth=0.5)

    # Add checkmarks
    for i in range(n):
        for j in range(n):
            text = ax2.text(j, i, 'Y' if attn_mask[i, j] else 'X',
                          ha='center', va='center',
                          color='darkgreen' if attn_mask[i, j] else 'darkred',
                          fontsize=10, fontweight='bold')

    # Add legend
    ax2.text(n+0.5, 1, 'Y = Can attend',
             ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax2.text(n+0.5, 3, 'X = Cannot attend',
             ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

    # Add region annotations
    # Prefix block
    ax2.add_patch(Rectangle((-0.5, -0.5), 5, 5,
                            fill=False, edgecolor=COLORS['attention'],
                            linewidth=3, linestyle='--'))
    ax2.text(2, -1, 'Prefix Block\n(Fully connected)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color=COLORS['attention'])

    # Suffix block
    ax2.add_patch(Rectangle((4.5, 4.5), 4, 4,
                            fill=False, edgecolor=COLORS['action'],
                            linewidth=3, linestyle='--'))
    ax2.text(6.5, n, 'Suffix Block\n(Causal)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color=COLORS['action'])

    plt.tight_layout()
    plt.savefig('attention_mask_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Attention mask visualization saved: attention_mask_visualization.png")
    plt.close()


def create_adarms_diagram():
    """Create AdaRMS mechanism diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    ax.text(5, 11.5, 'AdaRMS Mechanism (Pi0.5)',
            ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(5, 11, 'Adaptive RMS Normalization',
            ha='center', va='top', fontsize=14, color='gray')

    y = 10

    # Input
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

    # RMS computation
    rms_box = FancyBboxPatch((0.5, y-1), 4, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor='#3498db', alpha=0.3,
                             edgecolor='#2980b9', linewidth=2)
    ax.add_patch(rms_box)
    ax.text(2.5, y-0.3, 'Compute RMS',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.5, y-0.65, 'rms = sqrt(mean(x^2) + epsilon)',
            ha='center', va='center', fontsize=9)
    ax.text(2.5, y-0.9, 'x_norm = x / rms',
            ha='center', va='center', fontsize=9)

    # Condition projection
    proj_box = FancyBboxPatch((5.5, y-1), 4, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor='#1abc9c', alpha=0.3,
                              edgecolor='#16a085', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(7.5, y-0.3, 'Adaptive Scaling',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.5, y-0.55, 'Linear(cond_dim -> D)',
            ha='center', va='center', fontsize=9)
    ax.text(7.5, y-0.75, 'v',
            ha='center', va='center', fontsize=12)
    ax.text(7.5, y-0.9, 'alpha = MLP(time_emb)',
            ha='center', va='center', fontsize=9)

    y -= 1.5

    # Standard scaling
    gamma_box = FancyBboxPatch((1, y-0.6), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#f39c12', alpha=0.3,
                               edgecolor='#e67e22', linewidth=2)
    ax.add_patch(gamma_box)
    ax.text(2.5, y-0.35, 'Learned Scale gamma\n[D]',
            ha='center', va='center', fontsize=10)

    alpha_box = FancyBboxPatch((6, y-0.6), 3, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#9b59b6', alpha=0.3,
                               edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(alpha_box)
    ax.text(7.5, y-0.35, 'Adaptive alpha\n[B, 1, D]',
            ha='center', va='center', fontsize=10)

    y -= 1.5

    # Combination
    combine_box = FancyBboxPatch((1.5, y-1), 7, 0.9,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#e74c3c', alpha=0.3,
                                 edgecolor='#c0392b', linewidth=3)
    ax.add_patch(combine_box)
    ax.text(5, y-0.3, 'Adaptive Normalization',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y-0.65, 'output = x_norm * (gamma + alpha)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    y -= 1.5

    # Output
    output_box = FancyBboxPatch((2.5, y-0.8), 5, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#2ecc71', alpha=0.5,
                                edgecolor='#27ae60', linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, y-0.55, 'Output: Normalized Features\n[B, L, D]',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#34495e')

    ax.annotate('', xy=(2.5, 8.5), xytext=(2.5, 9.4), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 8.5), xytext=(7.5, 9.4), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 6.5), xytext=(2.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 6.5), xytext=(7.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.5), xytext=(2.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.5), xytext=(7.5, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.6), arrowprops=arrow_props)

    # Add advantages description
    ax.text(5, 1.5, 'Advantages:',
            ha='center', va='center', fontsize=12, fontweight='bold')

    advantages = [
        '1. Deep integration of time information into every layer',
        '2. Dynamic normalization strength adjustment',
        '3. Improved gradient flow',
        '4. Better conditional generation quality'
    ]

    for i, adv in enumerate(advantages):
        ax.text(5, 1.1 - i*0.3, adv,
                ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('adarms_mechanism.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ AdaRMS mechanism saved: adarms_mechanism.png")
    plt.close()


def create_model_comparison():
    """Create model comparison diagram"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    models = ['Pi0', 'Pi0-Fast', 'Pi0.5']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (ax, model, color) in enumerate(zip(axes, models, colors)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(5, 9.5, model,
                ha='center', va='top', fontsize=16, fontweight='bold', color=color)

        y = 8.5

        if model == 'Pi0':
            # Pi0 features
            features = [
                ('Action Generation', 'Diffusion Model'),
                ('Time Embedding', 'MLP Mixing'),
                ('State Processing', 'Separate Token'),
                ('Normalization', 'Standard RMSNorm'),
                ('Inference Speed', 'Slow (10 ODE steps)')
            ]
        elif model == 'Pi0-Fast':
            features = [
                ('Action Generation', 'Autoregressive'),
                ('Tokenization', 'VQ Codebook'),
                ('State Processing', 'Fused in Prefix'),
                ('Training', 'Cross Entropy Loss'),
                ('Inference Speed', 'Fast (1 step)')
            ]
        else:  # Pi0.5
            features = [
                ('Action Generation', 'Diffusion Model'),
                ('Time Embedding', 'AdaRMS Conditioning'),
                ('State Processing', 'No Separate Token'),
                ('Normalization', 'AdaRMS'),
                ('Inference Speed', 'Slow (10 ODE steps)')
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

    plt.suptitle('OpenPI Model Variants Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Model comparison saved: model_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("\n=== Generating OpenPI Architecture Visualizations ===\n")

    create_pi0_architecture()
    create_flow_matching_diagram()
    create_attention_mask_diagram()
    create_adarms_diagram()
    create_model_comparison()

    print("\n=== All diagrams generated successfully! ===")
    print("\nGenerated files:")
    print("1. pi0_architecture.png - Complete Pi0 model architecture")
    print("2. flow_matching_process.png - Flow Matching training and inference process")
    print("3. attention_mask_visualization.png - Attention mask visualization")
    print("4. adarms_mechanism.png - AdaRMS mechanism details")
    print("5. model_comparison.png - Three model variants comparison")
    print("\nPlease check the current directory for PNG files.\n")
