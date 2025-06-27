# nanoGPT-DAG ✧ _lightweight numeric reasoning for tiny GPTs_

This fork drops a **differentiable directed-acyclic-graph (DAG) module** on top of
[nanoGPT](https://github.com/karpathy/nanoGPT).  
The goal is to give a very small language model a dedicated sub-network that can
**read numbers out of text, perform a few arithmetic steps, then fold the
result back into the token stream**.

---

## How it works — one-paragraph version

1. **Plain GPT** runs as usual up to the final layer-norm.  
2. Those hidden states are **split three ways**  

   | branch | job |
   |--------|-----|
   | *node-embed block* | turn each token hidden state into a richer **embedding** that will live in the DAG |
   | *value extractor*  | attend over that embedding and spit out a **scalar value** (trying to recover the number the token represents, if any) |
   | *operand / op ctx blocks* | compress the sequence into two context vectors that will help the DAG choose its inputs and operations |

3. The **DAG controller** repeatedly  
   * attends over previous nodes (embeddings *and* values),  
   * picks two of them and an operation (`+ × − ÷ log pow max min identity`),  
   * creates a new node (new value + new embedding) and appends it.

4. After *k* steps we project **all DAG values back to embeddings**, do one
   self-attention + mean-pool, clean it with a tiny transformer block, and get a
   single vector `dag_sem`.

5. A learned **gate** mixes `dag_sem` with the last token's hidden state, then
   the usual `lm_head` produces logits.

---

## DAG Computation Process

The DAG module processes tokens sequentially, maintaining strict causality where each token can only access information from previous tokens. Here's how it works:

```mermaid
flowchart TD
    subgraph "Token Position t=2"
        direction TB
        T2["Token 2 Hidden State"] --> N2["Initial Node Embedding"]
        N2 --> V2["Initial Value"]
        subgraph "DAG Step"
            direction TB
            P2["Previous Nodes (t=0,1)"] --> F2["Flatten All Nodes"]
            F2 --> S2["Select Operands"]
            S2 --> O2["Apply Operation"]
            O2 --> NN2["New Node"]
        end
    end

    subgraph "Token Position t=1"
        direction TB
        T1["Token 1 Hidden State"] --> N1["Initial Node Embedding"]
        N1 --> V1["Initial Value"]
        subgraph "DAG Step"
            direction TB
            P1["Previous Nodes (t=0)"] --> F1["Flatten All Nodes"]
            F1 --> S1["Select Operands"]
            S1 --> O1["Apply Operation"]
            O1 --> NN1["New Node"]
        end
    end

    subgraph "Token Position t=0"
        direction TB
        T0["Token 0 Hidden State"] --> N0["Initial Node Embedding"]
        N0 --> V0["Initial Value"]
        subgraph "DAG Step"
            direction TB
            Z0["No Previous Nodes"] --> I0["Initialize with Zeros"]
            I0 --> NN0["New Node"]
        end
    end

    %% Node Access Patterns
    NN0 --> P1
    N0 --> P1
    NN1 --> P2
    N1 --> P2
    N0 --> P2

    %% Styling
    classDef token fill:#f9f,stroke:#333,stroke-width:2px
    classDef node fill:#bbf,stroke:#333,stroke-width:2px
    classDef step fill:#dfd,stroke:#333,stroke-width:2px
    class T0,T1,T2 token
    class N0,N1,N2,NN0,NN1,NN2 node
    class P1,P2,F1,F2,S1,S2,O1,O2 step
```

### Key Components:

1. **Initial Node Creation**:
   - Each token's hidden state is transformed into an initial node embedding
   - A value extractor converts this embedding into a scalar value
   - These form the initial node for that token position

2. **Node Storage**:
   - Nodes are stored in a (B, N, T, H) tensor where:
     - B: batch size
     - N: number of nodes per token (dag_depth + 1)
     - T: sequence length
     - H: hidden dimension

3. **Causal Processing**:
   - For token position t=0:
     - No previous nodes available
     - Initialize with zeros
   - For token position t>0:
     - Access ALL nodes from ALL previous tokens
     - Flatten these nodes for selection
     - Controller selects operands and operation
     - Create new node and add to position t

4. **Node Selection**:
   - Controller uses attention to select operands from previous tokens
   - Each token has access to all nodes (initial + computed) from previous tokens
   - Selection is guided by operand and operation context vectors

This structure ensures that:
- Each token can only access information from previous tokens
- All nodes from previous tokens are available for computation
- The DAG grows naturally with sequence length
- Causality is maintained through data structure rather than masking

---

## Architecture diagram

```mermaid
flowchart TD
    A["Input tokens  (B × T)"] --> B["Token ⨁ Pos embed  (B × T × H)"]
    B --> C["GPT blocks"]

    %% branches that prepare DAG inputs
    C --> D["Node-embed block ▶ node_embeds (B × T × H)"]
    D --> E["Value extractor ▶ values (B × T)"]

    %% operand / operator contexts
    C --> F["Operand-ctx block ▶ (B × H)"]
    C --> G["Op-ctx block ▶ (B × H)"]

    %% DAG
    subgraph Differentiable DAG
        direction TB
        H["Initial DAG nodes ⟨embeds, values⟩"] --> I["DAG controller + k steps"]
        F --> I
        G --> I
    end

    %% connect embeddings and values to DAG input
    D -->|embeds| H
    E -->|values| H

    %% post-DAG aggregation
    I --> J["Scalar → Embed proj"]
    J --> K["Value self-attn + pool ▶ (B × H)"]
    K --> L["Post-DAG block ▶ dag_sem"]

    %% fuse with transformer stream
    C --> M["Gate (sigmoid mix)"]
    L --> M
    M --> N["LM head ▶ logits (B × T × |V|)"]
```

---

## Install

```bash
pip install -r requirements-dev.txt
```

---

## Quick run (CPU toy)

```bash
python train.py config/train_default.py --dag_depth=4
```

Any field in `TrainConfig` can be overridden, e.g.
`--batch_size=4 --max_iters=100`.

---

## Tests

```bash
pytest
```

Tests cover:

* arithmetic op helpers  
* DAG growth & gradients  
* `extra_vals()` diagnostic hook  
* training-loop integration.

---

## Cloud training on RunPod

```bash
python3 -m venv env && source env/bin/activate
pip install requests runpod
export RUNPOD_API_KEY=<your_key>      # RunPod
export WANDB_API_KEY=<your_wandb_key> # Weights & Biases
python runpod_service.py train config/train_default.py --gpu-type "NVIDIA A100-40GB"
```

The helper prints the pod-id, GPU type and a direct W&B link so you can watch
loss curves live.

---

## Why learn the number extractor?

* **Multi-token numerals** – "12 345" is often `Ġ12`, `Ġ34`, `Ġ5`.  
* **Context disambiguation** – "route 66" ≠ numeric `66`.  
* **Unit scaling** – learn that "3 k" ⇒ `3000`.  
* **End-to-end gradients** – the extractor and embeddings co-adapt.
