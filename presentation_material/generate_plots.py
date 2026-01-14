import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

# Create directory for plots
os.makedirs("presentation_plots", exist_ok=True)

# Colors
BLUE = '#007BFF'    # Head
ORANGE = '#FF7F0E'  # Tail
GREEN = '#2CA02C'   # Relation/Logic
GRAY = '#CCCCCC'    # Grid/Faded

def setup_fig(figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)  # Transparent background
    ax.patch.set_alpha(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRAY)
    ax.spines['bottom'].set_color(GRAY)
    ax.tick_params(colors='#888888')
    ax.grid(True, linestyle='--', alpha=0.3, color=GRAY)
    return fig, ax

# Slide 1: TransE - showing 1-to-N limitation
def plot_transe():
    fig, ax = setup_fig()
    
    # Atlantic Salmon (head)
    h = [2, 4]
    ax.scatter(*h, color=BLUE, s=400, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate('Atlantic\nSalmon', h, xytext=(-30, 20), textcoords='offset points', 
                color=BLUE, fontsize=11, fontweight='bold', ha='center')
    
    # Two treatments (tails)
    t1, t2 = [6, 5], [6, 3]
    ax.scatter(*t1, color=ORANGE, s=300, zorder=5, edgecolors='black', linewidth=2)
    ax.scatter(*t2, color=ORANGE, s=300, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate('H₂O₂', t1, xytext=(15, 10), textcoords='offset points', 
                color=ORANGE, fontsize=10, fontweight='bold')
    ax.annotate('Azamethiphos', t2, xytext=(15, -5), textcoords='offset points', 
                color=ORANGE, fontsize=10, fontweight='bold')
    
    # Relation arrows
    ax.arrow(h[0], h[1], t1[0]-h[0]-0.3, t1[1]-h[1]-0.1, head_width=0.15, head_length=0.2, 
             fc=GREEN, ec=GREEN, length_includes_head=True, zorder=4, linewidth=2.5)
    ax.arrow(h[0], h[1], t2[0]-h[0]-0.3, t2[1]-h[1]+0.1, head_width=0.15, head_length=0.2, 
             fc='red', ec='red', length_includes_head=True, zorder=4, linewidth=2.5, linestyle='dashed', alpha=0.6)
    
    ax.text(4, 4.8, 'treats_with', color=GREEN, fontsize=12, fontweight='bold')
    ax.text(4, 3.2, 'fails!', color='red', fontsize=11, fontweight='bold', style='italic')
    
    # Limitation text
    ax.text(2, 6, 'TransE Problem: h + r ≠ t₁ and h + r ≠ t₂', 
            color='#333333', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlim(1, 7.5); ax.set_ylim(2, 6.5)
    ax.set_xlabel('Embedding Dimension 1', fontsize=9, color='#666666')
    ax.set_ylabel('Embedding Dimension 2', fontsize=9, color='#666666')
    plt.tight_layout()
    plt.savefig('presentation_plots/slide1_transe.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()

# Slide 2: RotatE - rotation for handling multiple relations
def plot_rotate():
    fig = plt.figure(figsize=(7, 7))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection='polar')
    ax.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    # Atlantic Salmon (center entity)
    ax.scatter(0, 1.2, color=BLUE, s=500, zorder=5, edgecolors='black', linewidth=2)
    
    # Multiple spawning locations at different angles
    angles = [np.pi/6, np.pi/2, 5*np.pi/6]
    locations = ['Norwegian\nFjord', 'Scottish\nRiver', 'Canadian\nStream']
    colors_loc = [ORANGE, ORANGE, ORANGE]
    
    for i, (angle, loc) in enumerate(zip(angles, locations)):
        ax.scatter(angle, 1.2, color=colors_loc[i], s=350, zorder=5, edgecolors='black', linewidth=2)
        # Rotation arc
        theta = np.linspace(0, angle, 50)
        ax.plot(theta, [1.2]*50, color=GREEN, linewidth=3.5, alpha=0.7, zorder=3)
        
    # Labels
    ax.text(0, 1.45, 'Atlantic\nSalmon', color=BLUE, fontweight='bold', fontsize=11, ha='center')
    ax.text(np.pi/6, 1.5, locations[0], color=ORANGE, fontweight='bold', fontsize=9, ha='center')
    ax.text(np.pi/2, 1.5, locations[1], color=ORANGE, fontweight='bold', fontsize=9, ha='center')
    ax.text(5*np.pi/6, 1.5, locations[2], color=ORANGE, fontweight='bold', fontsize=9, ha='center')
    
    # Rotation explanation
    ax.text(np.pi/4, 0.6, 'Different\nRotations', color=GREEN, fontsize=11, 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_rticks([1.2])
    ax.set_yticklabels([])
    ax.set_theta_zero_location('E')
    ax.grid(True, alpha=0.3, color=GRAY)
    plt.tight_layout()
    plt.savefig('presentation_plots/slide2_rotate.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()

# Slide 3: DistMult (Symmetric relations)
def plot_distmult():
    fig, ax = setup_fig()
    ax.axvline(x=0, color=GREEN, linestyle='--', linewidth=3, alpha=0.4, label='Symmetry Axis')
    
    # Symmetric relation: co-occurs_with
    p1, p2 = [-2.5, 3], [2.5, 3]
    ax.scatter(*p1, color=BLUE, s=400, zorder=5, edgecolors='black', linewidth=2)
    ax.scatter(*p2, color=BLUE, s=400, zorder=5, edgecolors='black', linewidth=2)
    
    ax.annotate('Lice_Strain_A', p1, xytext=(-15, 20), textcoords='offset points', 
                color=BLUE, fontweight='bold', fontsize=10, ha='center')
    ax.annotate('Lice_Strain_B', p2, xytext=(15, 20), textcoords='offset points', 
                color=BLUE, fontweight='bold', fontsize=10, ha='center')
    
    # Bidirectional arrows
    ax.annotate('', xy=(p2[0]-0.3, p2[1]), xytext=(p1[0]+0.3, p1[1]),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.annotate('', xy=(p1[0]+0.3, p1[1]-0.2), xytext=(p2[0]-0.3, p2[1]-0.2),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    
    ax.text(0, 3.8, 'co-occurs_with', color=GREEN, fontweight='bold', fontsize=11, ha='center')
    ax.text(0, 0.5, 'DistMult: Symmetric Only', color='#333333', rotation=270, 
            fontweight='bold', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlim(-4, 4); ax.set_ylim(0, 5)
    ax.set_xlabel('Embedding Dimension 1', fontsize=9, color='#666666')
    ax.set_ylabel('Embedding Dimension 2', fontsize=9, color='#666666')
    plt.tight_layout()
    plt.savefig('presentation_plots/slide3_distmult.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()

# Slide 4: ComplEx (Asymmetric relations)
def plot_complex():
    fig, ax = setup_fig()
    
    h, t = [1.5, 3], [5.5, 3]
    ax.scatter(*h, color=BLUE, s=450, zorder=5, edgecolors='black', linewidth=2)
    ax.scatter(*t, color=ORANGE, s=450, zorder=5, edgecolors='black', linewidth=2)
    
    # Forward path (resistant_to)
    ax.arrow(h[0], h[1]+0.3, t[0]-h[0]-0.5, 0, head_width=0.2, head_length=0.3, 
             fc=GREEN, ec=GREEN, length_includes_head=True, linewidth=3, zorder=4)
    
    # Backward path (Blocked)
    ax.plot([t[0]-0.2, h[0]+0.2], [t[1]-0.3, h[1]-0.3], 'r--', linewidth=3, alpha=0.7)
    ax.text(3.5, 2.3, '✗ NOT symmetric', color='red', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate('Salmon\nCoho', h, xytext=(-10, 25), textcoords='offset points', 
                color=BLUE, fontweight='bold', fontsize=11, ha='center')
    ax.annotate('Lice_Strain\nResistant', t, xytext=(10, 25), textcoords='offset points', 
                color=ORANGE, fontweight='bold', fontsize=10, ha='center')
    
    ax.text(3.5, 3.7, 'resistant_to', color=GREEN, fontweight='bold', ha='center', fontsize=12)
    
    # Advantage text
    ax.text(3.5, 1, 'ComplEx: Handles directional relations!', 
            color='#333333', fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))
    
    ax.set_xlim(0, 7); ax.set_ylim(0.5, 4.5)
    ax.set_xlabel('Real Component', fontsize=9, color='#666666')
    ax.set_ylabel('Imaginary Component', fontsize=9, color='#666666')
    plt.tight_layout()
    plt.savefig('presentation_plots/slide4_complex.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()

# Slide 8: Treatment Recommendation
def plot_recommendation():
    fig, ax = setup_fig(figsize=(7, 7))
    center = [0, 0]
    
    # Query: Salmon farm with lice outbreak
    ax.scatter(*center, color='red', s=600, zorder=10, marker='*', edgecolors='black', linewidth=2)
    ax.text(0, 0.4, 'Farm XYZ\n(Lice Outbreak)', color='red', ha='center', fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Recommended treatments (neighbors)
    np.random.seed(42)
    treatments = ['H₂O₂\n(87%)', 'Cleaner\nFish (92%)', 'Azamethiphos\n(78%)', 
                  'Thermal\nTreatment (85%)', 'Mechanical\nRemoval (81%)', 'Deltamethrin\n(73%)']
    angles = np.linspace(0, 2*np.pi, len(treatments), endpoint=False)
    
    for i, (angle, treatment) in enumerate(zip(angles, treatments)):
        r = 2.2 + np.random.normal(0, 0.15)
        p = [r*np.cos(angle), r*np.sin(angle)]
        
        # Color by efficacy
        efficacy = float(treatment.split('(')[1].strip('%)')) if '(' in treatment else 80
        if efficacy > 85:
            color = GREEN
            size = 400
        elif efficacy > 80:
            color = ORANGE
            size = 350
        else:
            color = '#FF6B6B'
            size = 300
            
        ax.scatter(*p, color=color, s=size, alpha=0.8, zorder=5, edgecolors='black', linewidth=1.5)
        ax.plot([0, p[0]], [0, p[1]], color=GRAY, alpha=0.3, linestyle=':', linewidth=2, zorder=1)
        
        # Treatment labels
        label_offset = 1.15
        ax.text(p[0]*label_offset, p[1]*label_offset, treatment, 
                ha='center', va='center', fontsize=8, fontweight='bold', color='#333333')
    
    # Legend
    ax.text(0, -3.5, 'KG-based Treatment Ranking', color='#555555', ha='center', 
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig('presentation_plots/slide8_recommendation.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Generating Salmon Lice Treatment KG Visualization Plots")
    print("="*60 + "\n")
    
    plots = [
        ("TransE (1-to-N limitation)", plot_transe),
        ("RotatE (Rotation for multiple relations)", plot_rotate),
        ("DistMult (Symmetric relations)", plot_distmult),
        ("ComplEx (Asymmetric relations)", plot_complex),
        ("Treatment Recommendation", plot_recommendation)
    ]
    
    for name, func in plots:
        print(f"  ✓ Generating: {name}...")
        func()
    
    print("\n" + "="*60)
    print(f" ✓ SUCCESS: All plots saved to 'presentation_plots/' folder")
    print("="*60 + "\n")
