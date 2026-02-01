The Problem
Standard GNNs give all nodes the same embedding dimension. But some nodes need more capacity (bottleneck nodes, high-degree nodes) while others need less. Fixed dimensions waste parameters.

The Core Idea
Adaptive capacity allocation: Give each node a personalized embedding dimension based on its structural importance, but do it efficiently by sharing parameters across capacity levels.

How It Works (4 Steps)
1. Compute Structural Features (lines 333-388)
For each node, compute 3 features that indicate capacity needs:

Degree: High-degree nodes aggregate many messages → need more capacity
Neighbor degree variance: Nodes connecting high/low degree regions (bottlenecks) → need more capacity
Bridge score: Nodes with low clustering connect different communities → need more capacity
2. Predict Capacity Assignment (CapacityPredictor, lines 391-418)
A small MLP looks at those 3 structural features and outputs soft weights over K capacity "bins":

Bin 0: Smallest dimension (e.g., 128)
Bin 1: Medium-small (e.g., 160)
Bin 2: Medium-large (e.g., 192)
Bin 3: Largest (e.g., 256)
Each node gets a soft assignment like [0.1, 0.3, 0.4, 0.2] — mostly using bins 2-3 but some from 1.

3. Parallel Processing in Multiple Dimensions (CapacityAwareConv, lines 421-529)
Here's the "shared backbone" part:

Create K separate GNN layers (one per bin), each with different dimensions
For each node's features:
Project to each bin's dimension
Process through that bin's GNN layer
Project back to base dimension
Weight by the node's soft assignment to that bin
Sum all weighted outputs
So if a node has weights [0.1, 0.3, 0.4, 0.2]:

4. Message Passing (CapacityAwareGNNEncoder, lines 532-577)
Apply this capacity-aware convolution for multiple layers with:

Residual connections
Layer normalization
The same capacity predictor across all layers (structural features computed once)
Why "Shared Backbone"?
Shared: All nodes use the same set of K GNN layers (shared parameters)
Backbone: These K layers form the backbone that's mixed differently per node
Efficient: Instead of having personalized dimensions per node (millions of parameters), you have K shared layers that get combined with different weights

Here are 2-minute breakdowns of **GOKU (Spectrum-Preserving Sparsification)** and **SDRF (Spectral Distance Rewiring)**:

---

## **GOKU (Spectrum-Preserving Sparsification)**

### The Problem
Dense graphs are computationally expensive and can cause over-smoothing. But naively removing edges breaks the graph's spectral properties (eigenvalues), which encode important structural information that GNNs rely on.

### The Core Idea
**Remove edges smartly**: Keep only the most important edges that preserve the graph's spectrum (Laplacian eigenvalues), so the graph "looks the same" to a GNN but has fewer edges.

### How It Works (3 Steps)

**1. Compute Effective Resistance** (lines 603-644)
For each edge (u,v), compute its **effective resistance** using the spectral formula:

$$R_{uv} = \sum_{i=2}^{k} \frac{1}{\lambda_i} (\phi_i(u) - \phi_i(v))^2$$

Where:
- $\lambda_i$ = Laplacian eigenvalues (small eigenvalues → high importance)
- $\phi_i$ = eigenvectors
- High resistance = edge is a **bottleneck** (critical for connectivity)

**2. Probabilistic Sampling**
- Convert resistances to probabilities (higher resistance → higher probability)
- Sample edges to keep based on these probabilities
- Keep `sparsification_ratio` fraction (e.g., 50% of edges)

**3. Preserve Spectrum**
The sampling is designed so that in expectation, the sparsified graph's Laplacian eigenvalues are close to the original. This means the graph's diffusion properties, bottlenecks, and community structure remain intact.

### Why It Works
- **Weighted by eigenvalues**: Small eigenvalues (global structure) get more weight than large ones (local noise)
- **Keeps bottlenecks**: High-resistance edges (bridges between communities) are prioritized
- **Spectral guarantee**: Preserves the graph's "essence" mathematically

---

## **SDRF (Spectral Distance Rewiring with Filtration)**

### The Problem
Over-squashing happens when information must travel far through bottlenecks. Standard GNNs suffer because nodes in the same community are often far apart in the original graph.

### The Core Idea
**Add shortcuts between similar nodes**: Create edges between nodes that are **spectrally close** (structurally similar) even if they're far apart, while removing edges between spectrally distant nodes.

### How It Works (4 Steps)

**1. Spectral Embedding** (lines 678-710)
- Compute Laplacian eigenvectors (skip the first constant one)
- Each node gets coordinates in k-dimensional spectral space
- Nodes in the same eigenvector coordinates → structurally similar roles

**2. Compute Spectral Distances** (lines 693-696)
For all node pairs (i,j):
$$d_{ij} = \|\phi_i - \phi_j\|_2$$

This is just **Euclidean distance in the spectral embedding space** (unweighted).

**3. Iterative Rewiring** (lines 712-751)
Repeat for multiple iterations:
- **Remove**: Delete edges where $d_{ij} >$ `removal_bound` (spectrally distant)
- **Add**: Connect each node to its k-nearest neighbors in spectral space

**4. Progressive Improvement**
- After removing/adding edges, recompute spectral distances on the new graph
- The graph topology evolves to better reflect spectral similarity
- Convergence after several iterations

### Why It Works
- **Community-aware**: Nodes in the same community have similar eigenvector coordinates
- **Bottleneck reduction**: Creates shortcuts across long paths in the original graph
- **Unweighted distance**: Treats all eigenvectors equally (unlike GOKU which weights by eigenvalues)

---

## **Key Differences: GOKU vs SDRF**

| Aspect | GOKU (SPS) | SDRF |
|--------|------------|------|
| **Operation** | Sparsification (removes edges) | Rewiring (removes + adds) |
| **Distance metric** | Effective resistance (eigenvalue-weighted) | Spectral distance (unweighted L2) |
| **Formula** | $R_{uv} = \sum \frac{1}{\lambda_i}(\phi_i(u)-\phi_i(v))^2$ | $d_{uv} = \|\phi_u - \phi_v\|_2$ |
| **Goal** | Preserve spectrum while reducing edges | Create shortcuts between similar nodes |
| **Edge strategy** | Probabilistic sampling based on resistance | k-NN in spectral space + remove distant |
| **Iteration** | One-shot operation | Iterative refinement |
| **Eigenvalue weight** | Yes (1/λ weights small eigenvalues more) | No (equal weight to all eigenvectors) |

---

**Bottom line**: 
- **GOKU** says: "Keep only edges that are critical for the graph's spectral properties"
- **SDRF** says: "Connect nodes that are structurally similar, regardless of original distance"

Both use spectral analysis but with different philosophies: GOKU is conservative (preserve what exists), SDRF is transformative (reshape the graph).

