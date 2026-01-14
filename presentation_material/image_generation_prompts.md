# Visual Storytelling Guide: Knowledge Graph Embeddings (Prompts)

As a Senior UX Designer, I have crafted these prompts to ensure a cohesive visual narrative for your presentation. We use a **consistent color language** to help the audience immediately recognize the components across different mathematical paradigms.

## 🎨 Visual Language System (The Branding)
*   **Head Entity (h):** Electric Blue Sphere (`#007BFF`)
*   **Tail Entity (t):** Neon Orange Sphere (`#FF7F0E`)
*   **Relation (r):** Vibrant Emerald Green Arrow/Path (`#2CA02C`)
*   **Environment:** Pure White Background with minimalist, light-gray thin grid lines. No dark shading.

---

## Slide 1: TransE - The Translation Vector
**Logic:** $h + r \approx t$. Euclidean distance based.
**Prompt:**
> *Professional 2D vector graphic on a pure white background with thin, minimalist light-gray grid lines. Electric Blue sphere ('Salmon') at [2, 4]. Neon Orange sphere ('Fish') at [5, 3]. A Vibrant Emerald Green arrow (+3, -1) connects them. Labeled 'is_a'. Text overlay: 'h + r = t'. Minimalist, clean, high-contrast flat design style.*

## Slide 2: RotatE - Hadamard Product in Complex Space
**Logic:** $t = h \circ r$, where $|r_i|=1$. Captures symmetry and inversion.
**Prompt:**
> *Professional 2D polar coordinate system (Complex Plane) on a pure white background. Electric Blue sphere ('Atlantic Salmon') at 0°. Neon Orange sphere ('Spawning Ground') at 90°. A curved Vibrant Emerald Green arc shows the rotation. Labeled 'r = e^(iθ)'. Captures 'Return Migration' flow. Clean lines, minimalist aesthetic, no dark shadows.*

## Slide 3: DistMult - Symmetric Scaling
**Logic:** $h^\top \text{diag}(r) t$. Symmetric scoring function.
**Prompt:**
> *2D Grid visualization on a pure white background with thin gray grid lines. Electric Blue sphere ('Shark') and Neon Orange sphere ('Tuna') are equidistant from a central Vibrant Emerald Green 'Scaling Mirror'. Reflective lines show they are perfectly symmetric. Labeled 'co-exists_with'. Illustrates the symmetry of DistMult. Flat vector design style.*

## Slide 4: ComplEx - Hermitian Inner Product
**Logic:** Captures asymmetry using Real and Imaginary components.
**Prompt:**
> *2D split-view (Real vs Imaginary) on a pure white background. A Vibrant Emerald Green 'directional lens' labeled 'preys_on'. It shows an Electric Blue sphere ('Orca') passing through and hitting a Neon Orange sphere ('Seal'), but the reverse path is blocked/mismatched. Visualizing the ability to handle asymmetric fish food chains. Minimalist flat design.*

## Slide 5: AutoSF - Fragment Search
**Logic:** Discovered optimal scoring patterns via AutoML.
**Prompt:**
> *2D mosaic style on a pure white background. Various Vibrant Emerald Green structural 'puzzle pieces' of different shapes are being tested to connect an Electric Blue sphere ('Coral') to a Neon Orange sphere ('Reef Fish'). One piece fits perfectly. Labeled 'Automated Structure Matching'. Clean, professional modern aesthetic.*

## Slide 6: Transformer Baseline - Global Attention
**Logic:** All-to-all connectivity via Multi-Head Attention.
**Prompt:**
> *2D network graph on a pure white background. An Electric Blue sphere is connected by dozens of thin, Vibrant Emerald Green 'energy threads' (attention heads) to every other entity in the scene (Crabs, Kelp, Shells, Currents). Highlights everything is connected to everything. Labeled 'Global Semantic Context'. Clean, geometric style.*

## Slide 7: SSM (Our Model) - Selective State Flow
**Logic:** Linear recurrence and state transitions. $h_t = Ah_{t-1} + Bx_t$.
**Prompt:**
> *2D flow-line on a white grid. An Electric Blue vector ('Tide') enters a Vibrant Emerald Green tunnel. Inside the tunnel, it changes state sequentially (State 1 -> State 2) before emerging as a prediction for the Neon Orange sphere ('Fish Location'). Visualizes 'Temporal/Causal Reasoning'. Minimalist vector art, white background.*

## Slide 8: Recommendation Discovery
**Concept:** Top-K candidates in the manifold.
**Prompt:**
> *A 2D scatter plot manifold on a pure white background. One Electric Blue sphere ('Trout') in the center. A ring of Vibrant Neon Orange spheres ('Fly Fishing', 'Freshwater', 'Stream') surrounds it, connected by soft Emerald Green gradients. Representing the final output of the recommendation engine. Minimalist and clean data visualization.*

---

## 🛠️ Usage Tips for the Slides:
1.  **Transition:** Use a "morph" transition between slides so the colors remain static while the "math" (arrows, arcs, lenses) changes.
2.  **Consistency:** Ensure the head entity is **always** on the left or top to reinforce the "Head -> Relation -> Tail" reading direction.
3.  **Contrast:** The dark background is intentional to make the Electric Blue and Neon Orange "pop," symbolizing "Knowledge shining in the dark."
