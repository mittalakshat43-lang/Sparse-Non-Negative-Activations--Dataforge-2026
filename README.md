# Explaining the Frontier: Adaptive Sparsity and Non-Negative Activations

**DataForge 2026, Pathway Track submission**
**Topic:** Sparse Non-Negative Activations

- Live artifact: https://mittalakshat43-lang.github.io/Sparse-Non-Negative-Activations--Dataforge-2026/
- Source repository: https://github.com/mittalakshat43-lang/Sparse-Non-Negative-Activations--Dataforge-2026
- One-page concept summary: `one_page_concept_summary.pdf` (in this repo)

## The claim

A network of non-negative, sparsely-firing units can represent information using only a small, variable fraction of its units at once. How many fire depends on how predictable the input is, not on a fixed quota.

## Who this is for

Machine learning students and data scientists who are moving between traditional dense Transformers and the newer brain-inspired, post-Transformer architectures.

**Prerequisites**
- A conceptual understanding of basic neural network anatomy: nodes, edges, layers
- Familiarity with standard activation functions, especially ReLU
- A basic grasp of what happens during a forward pass

**Learning objectives**

By the end, a learner should be able to:
- Tell apart polysemantic and monosemantic neuron behavior
- Separate structural sparsity (wires that are missing) from activation sparsity (signals that are silenced for a given input)
- See mathematically why a non-negative activation function is what makes true sparsity possible
- Watch a network adjust its own active-unit count based purely on how predictable its input is, echoing BDH's reported ~5% active rate

## What's in the artifact

The explainer is a single self-contained HTML page: seven numbered sections plus a closing recap. Open `index.html` directly in a browser; nothing needs to be installed to view it.

| Section | What it shows | How the learner interacts |
|---|---|---|
| 01. The Tangle Problem | Why dense networks end up polysemantic, one neuron standing for several unrelated concepts | Toggle between polysemantic and sparse modes, inspect a single neuron's activation trace |
| 02. The Structural Picture | The difference between a dense and a sparse network at the level of wiring | Slide the sparsity level and watch which connections drop out |
| 03. The Non-Negative Switch | What ReLU, Linear, and Sigmoid each do to a signal | Pick an activation function, drag the input value, read the output live |
| 04. Forcing Sparsity by Hand | What it looks like to prune a network by an externally set threshold, rather than let sparsity emerge from the input | Move a pruning threshold and watch nodes drop out of a fixed network |
| 05. The Real Thing | A genuine 200-unit network, trained in Python, predicting the next symbol in a sequence, live in the browser | Set how predictable the input sequence is, toggle a "force fixed quota" comparison against natural sparsity, generate new input windows |
| 06. The Layered Canvas | Why banning negative numbers also bans subtraction: a toy accumulation diagram, then a physical light-mixing analogy | Replay either animation on demand |
| 07. A New Kind of Engine | The three consequences of the constraint, pulled together into one synthesis | Read only, no control (by design) |
| Closing | A one-paragraph recap of the full arc | |

## What's live, precomputed, or illustrative

Being honest about this split matters for the "interactive substrate and honesty" scoring criterion, so it is stated plainly:

- **Live, real computation:** every control in Sections 01 through 05 recomputes from the actual input each time, including the activation-function curve in Section 03, the pruning statistics in Section 04, and, most importantly, Section 05's window generator, forward pass, prediction bars, and neuron grid, all of which run in the browser from weights trained in Python (see `train_sparse_net.py`).
- **Precomputed:** Section 05's reference table (expected active-unit percentage and expected confidence, by condition) was measured offline over 600+ trials per condition. It is not recomputed on page load; the page states this in its footer.
- **Illustrative, explicitly labeled on the page as such:** Section 06's accumulation diagram and light-beam analogy replay fixed, pre-set values to build intuition. They are not live model behavior, and the page says so directly next to each one.

## Reproducing the trained network (Section 05)

`train_sparse_net.py` trains the network on dynamically generated synthetic Markov chain sequences, using an L1 sparsity penalty to encourage sparse activations.

**Dependencies:** `numpy`

**Train the model:**
```
python train_sparse_net.py
```

**Export the weights into the HTML:** running the command above trains the model from scratch and automatically generates a local file named `sparse_net_weights.json`. To load these new weights into the web visualization, open `sparse_net_weights.json`, copy its entire contents, and paste it to replace the `const DATA = { ... };` object located inside the `<script>` block for "Section 5: Adaptive Sparsity" in `index.html`.

## Primary sources

Three recent primary papers ground the claims in this explainer, cited beside the specific claims they support inside the artifact and the one-page summary:

1. **Kosowski, A., Uznanski, P., Chorowski, J., Stamirowska, Z., and Bartoszkiewicz, M. (2025).** *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507. Source for BDH's architecture, its reported ~5% active-neuron figure, and its Hebbian synaptic update rule. Cited in Section 05.
2. **Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi, S. J., Ye, K., Chern, F., Yu, F., Guo, R., and Kumar, S. (2022).** *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers.* arXiv:2210.06313 (ICLR 2023). Independent evidence, unrelated to BDH, that ordinary ReLU-based Transformers develop activation sparsity as low as 3.0% nonzero without being designed to. Cited in Section 01.
3. **Mirzadeh, I., Alizadeh-Vahid, K., Mehta, S., Del Mundo, C. C., Tuzel, O., Samei, G., Rastegari, M., and Farajtabar, M. (2023).** *ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models.* arXiv:2310.04564 (ICLR 2024). Makes the case for deliberately reinstating ReLU in modern LLMs to reclaim activation sparsity for inference speedups. Cited in Section 07.

**On BDH-CQ:** Pathway's later reasoning system, BDH-CQ (*BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888), is not treated as load-bearing evidence for this concept. Its contribution is a recurrent contextual-memory and latent-reasoning mechanism for tasks like ARC-AGI, not the activation-level sparsity mechanism this artifact is about. It is mentioned in the one-page summary only to say so explicitly, rather than invent a connection that is not there.

## Source and license record

| Item | Source | License |
|---|---|---|
| Fraunces, IBM Plex Mono | Google Fonts | SIL Open Font License (OFL) |
| KaTeX | KaTeX project | MIT |
| Trained weights (Section 05) | Self-trained on a synthetic next-symbol-prediction task | Trained via the included `train_sparse_net.py` script using dynamically generated synthetic Markov chain sequences and an L1 sparsity penalty |
| Interactive JS widgets and network logic | Original work (vibe-coded) | MIT License |

## AI assistance disclosure

- **Artifact text content (HTML file):** Written entirely by us; we subsequently asked AI to refine the language and keep a more professional tone throughout.
- **Visuals:** The idea for each visual was entirely ours; we asked AI to code each visual exactly as we specified.
- **Final artifact compilation:** Compiled with the help of AI, after thorough review of its contents by us.
- **`train_sparse_net.py`:** Vibe-coded.
- **README:** Written entirely by us.
- **One-page concept summary:** Written entirely by us; we subsequently asked AI to refine the language and keep a professional tone.

## Credits

No external mentorship was sought.
