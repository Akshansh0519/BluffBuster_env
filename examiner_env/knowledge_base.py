"""
knowledge_base.py — Static ML-theory KB for The Examiner.

Provides mechanism cues, misconceptions, and probe templates for all 10
sections.  The oracle's LLR scorer reads from this module.

All imports are standard-library + pydantic only (no network calls).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# KB component schemas
# ──────────────────────────────────────────────────────────────────────────────

class MechanismCue(BaseModel):
    phrase: str
    cue_strength: Literal["weak", "strong"]

    @property
    def weight(self) -> float:
        return 1.0 if self.cue_strength == "strong" else 0.5


class MisconceptionPhrase(BaseModel):
    phrase: str
    severity: Literal["minor", "major"]

    @property
    def weight(self) -> float:
        return 1.0 if self.severity == "major" else 0.5


class ProbeTemplate(BaseModel):
    template: str
    probe_type: Literal[
        "definitional", "mechanism", "edge_case", "counterexample", "application"
    ]


class SectionKB(BaseModel):
    section_id: str
    title: str
    key_concepts: list[str] = Field(min_length=5)
    mechanism_cues: list[MechanismCue] = Field(min_length=3)
    common_misconceptions: list[MisconceptionPhrase] = Field(min_length=3)
    probe_templates: list[ProbeTemplate] = Field(min_length=3)
    evidence_weights: dict = Field(
        default_factory=lambda: {"alpha": 1.5, "beta": 0.5, "gamma": 1.0}
    )
    reference_responses: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helper to build the KB
# ──────────────────────────────────────────────────────────────────────────────

def build_kb() -> dict[str, SectionKB]:  # noqa: C901  (long but intentional)
    sections: list[SectionKB] = [

        # ── S01: Gradient Descent ─────────────────────────────────────────────
        SectionKB(
            section_id="S01",
            title="Gradient Descent and Optimization",
            key_concepts=[
                "learning rate", "gradient vector", "loss landscape",
                "convergence", "stochastic gradient descent", "momentum",
                "Adam optimizer", "saddle points",
            ],
            mechanism_cues=[
                MechanismCue(phrase="gradient of the loss with respect to parameters", cue_strength="strong"),
                MechanismCue(phrase="learning rate scales the magnitude of each update step", cue_strength="strong"),
                MechanismCue(phrase="momentum accumulates an exponentially decaying average of past gradients", cue_strength="strong"),
                MechanismCue(phrase="adaptive per-parameter learning rates in Adam via first and second moment estimates", cue_strength="strong"),
                MechanismCue(phrase="gradient clipping prevents exploding gradients by rescaling the gradient norm", cue_strength="strong"),
                MechanismCue(phrase="saddle point where gradient is zero but curvature is mixed", cue_strength="weak"),
                MechanismCue(phrase="stochastic noise from mini-batches helps escape sharp local minima", cue_strength="weak"),
                MechanismCue(phrase="learning rate schedule anneals step size during training", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="gradient descent always finds the global minimum", severity="major"),
                MisconceptionPhrase(phrase="a larger learning rate always trains the model faster without quality loss", severity="major"),
                MisconceptionPhrase(phrase="gradient descent only works on convex loss functions", severity="major"),
                MisconceptionPhrase(phrase="momentum just adds noise to help escape local minima", severity="minor"),
                MisconceptionPhrase(phrase="batch size has no effect on the final model quality", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is gradient descent?", probe_type="definitional"),
                ProbeTemplate(template="Why does gradient descent with momentum converge faster on ill-conditioned loss surfaces?", probe_type="mechanism"),
                ProbeTemplate(template="What happens when the learning rate is set too high in gradient descent?", probe_type="edge_case"),
                ProbeTemplate(template="Give a counterexample where gradient descent does not converge to the global optimum.", probe_type="counterexample"),
                ProbeTemplate(template="How would you apply Adam optimizer differently from vanilla SGD in practice?", probe_type="application"),
            ],
        ),

        # ── S02: Backpropagation ──────────────────────────────────────────────
        SectionKB(
            section_id="S02",
            title="Backpropagation",
            key_concepts=[
                "chain rule", "computational graph", "Jacobian",
                "upstream gradient", "vanishing gradient", "exploding gradient",
                "automatic differentiation", "reverse-mode AD",
            ],
            mechanism_cues=[
                MechanismCue(phrase="chain rule applied to the composition of differentiable functions", cue_strength="strong"),
                MechanismCue(phrase="Jacobian matrix of partial derivatives of layer outputs with respect to inputs", cue_strength="strong"),
                MechanismCue(phrase="upstream gradient from the next layer is multiplied element-wise with local gradient", cue_strength="strong"),
                MechanismCue(phrase="vanishing gradient occurs when sigmoid activations saturate and derivatives approach zero", cue_strength="strong"),
                MechanismCue(phrase="computational graph records operations during the forward pass for reverse-mode differentiation", cue_strength="strong"),
                MechanismCue(phrase="reverse-mode automatic differentiation traverses the graph from output to input", cue_strength="weak"),
                MechanismCue(phrase="weight gradient is the outer product of upstream gradient and layer input", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="backpropagation just reverses the computation of the forward pass", severity="major"),
                MisconceptionPhrase(phrase="each weight gradient is computed independently without sharing upstream signals", severity="major"),
                MisconceptionPhrase(phrase="backpropagation sends the actual error values backwards through the network", severity="major"),
                MisconceptionPhrase(phrase="deeper networks always have more vanishing gradient problems regardless of activation", severity="minor"),
                MisconceptionPhrase(phrase="backpropagation requires the loss function to be differentiable everywhere", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="Explain backpropagation.", probe_type="definitional"),
                ProbeTemplate(template="Why does backpropagation rely on the chain rule, and how does that chain decompose for a two-layer network?", probe_type="mechanism"),
                ProbeTemplate(template="What happens to gradients in very deep networks with sigmoid activations, and why?", probe_type="edge_case"),
                ProbeTemplate(template="Can you give an example where gradients explode rather than vanish, and what causes that?", probe_type="counterexample"),
                ProbeTemplate(template="How would you implement gradient accumulation using the computational graph concept?", probe_type="application"),
            ],
        ),

        # ── S03: Overfitting and Regularization ──────────────────────────────
        SectionKB(
            section_id="S03",
            title="Overfitting and Regularization",
            key_concepts=[
                "bias-variance tradeoff", "training error", "generalisation error",
                "L1 regularization", "L2 regularization", "dropout",
                "cross-validation", "early stopping",
            ],
            mechanism_cues=[
                MechanismCue(phrase="L2 regularisation adds a penalty proportional to the squared magnitude of weights encouraging smaller weights", cue_strength="strong"),
                MechanismCue(phrase="dropout randomly zeroes activations during training which approximates an ensemble of subnetworks", cue_strength="strong"),
                MechanismCue(phrase="bias-variance decomposition decomposes generalisation error into irreducible noise, bias, and variance terms", cue_strength="strong"),
                MechanismCue(phrase="cross-validation estimates generalisation error by averaging performance across held-out folds", cue_strength="strong"),
                MechanismCue(phrase="L1 regularisation induces sparsity by pushing small weights exactly to zero via subgradient", cue_strength="strong"),
                MechanismCue(phrase="early stopping uses validation loss as a proxy for generalisation to halt training before overfitting", cue_strength="weak"),
                MechanismCue(phrase="data augmentation effectively increases the training distribution without new labelled data", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="more training data always prevents overfitting regardless of model capacity", severity="major"),
                MisconceptionPhrase(phrase="regularisation just adds random noise to the training process", severity="major"),
                MisconceptionPhrase(phrase="a model with zero training error is always overfitting", severity="minor"),
                MisconceptionPhrase(phrase="dropout hurts performance at test time by randomly dropping information", severity="minor"),
                MisconceptionPhrase(phrase="L1 and L2 regularisation are interchangeable because they both penalise weight magnitude", severity="major"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is overfitting?", probe_type="definitional"),
                ProbeTemplate(template="How does L2 regularisation mechanistically prevent overfitting through the weight update rule?", probe_type="mechanism"),
                ProbeTemplate(template="When could adding more training data make overfitting worse?", probe_type="edge_case"),
                ProbeTemplate(template="Give a scenario where a heavily regularised model still overfits.", probe_type="counterexample"),
                ProbeTemplate(template="How would you choose between L1 and L2 regularisation for a feature selection problem?", probe_type="application"),
            ],
        ),

        # ── S04: Attention Mechanisms ─────────────────────────────────────────
        SectionKB(
            section_id="S04",
            title="Attention Mechanisms",
            key_concepts=[
                "query", "key", "value", "scaled dot-product attention",
                "softmax normalisation", "attention weights", "self-attention",
                "cross-attention",
            ],
            mechanism_cues=[
                MechanismCue(phrase="query-key dot product measures compatibility between a query vector and each key vector", cue_strength="strong"),
                MechanismCue(phrase="dividing by the square root of key dimension prevents softmax saturation in high dimensions", cue_strength="strong"),
                MechanismCue(phrase="softmax normalises scores into a probability distribution over value positions", cue_strength="strong"),
                MechanismCue(phrase="multi-head attention projects queries, keys, and values into multiple subspaces to capture diverse relationship patterns", cue_strength="strong"),
                MechanismCue(phrase="self-attention computes relationships between all pairs of positions within the same sequence", cue_strength="strong"),
                MechanismCue(phrase="attention is quadratic in sequence length because all pairwise scores must be computed", cue_strength="weak"),
                MechanismCue(phrase="causal masking in decoder prevents attending to future positions by setting those scores to negative infinity", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="attention just learns which tokens are important by assigning high weights to them", severity="major"),
                MisconceptionPhrase(phrase="attention weights always sum to exactly one because of softmax", severity="minor"),
                MisconceptionPhrase(phrase="multi-head attention is just running the same attention computation multiple times", severity="major"),
                MisconceptionPhrase(phrase="self-attention can only capture local context like convolutions do", severity="major"),
                MisconceptionPhrase(phrase="attention mechanisms require recurrence to model sequences", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is the scaled dot-product attention mechanism?", probe_type="definitional"),
                ProbeTemplate(template="Why do we scale by the square root of the key dimension in dot-product attention?", probe_type="mechanism"),
                ProbeTemplate(template="What happens when all query-key dot products are very similar in magnitude?", probe_type="edge_case"),
                ProbeTemplate(template="Give a case where multi-head attention would outperform single-head attention and explain why.", probe_type="counterexample"),
                ProbeTemplate(template="How would you modify attention for a task requiring the model not to see future tokens?", probe_type="application"),
            ],
        ),

        # ── S05: Transformer Architecture ─────────────────────────────────────
        SectionKB(
            section_id="S05",
            title="Transformer Architecture",
            key_concepts=[
                "positional encoding", "multi-head self-attention",
                "feed-forward sublayer", "residual connection",
                "layer normalisation", "encoder", "decoder",
                "pre-norm vs post-norm",
            ],
            mechanism_cues=[
                MechanismCue(phrase="positional encodings inject token position information because self-attention is permutation-invariant", cue_strength="strong"),
                MechanismCue(phrase="residual connections allow gradients to bypass attention sublayers preventing vanishing gradients in deep stacks", cue_strength="strong"),
                MechanismCue(phrase="layer normalisation normalises pre-activation statistics within each residual stream", cue_strength="strong"),
                MechanismCue(phrase="feed-forward sublayer projects to a higher-dimensional space and back introducing non-linearity after attention", cue_strength="strong"),
                MechanismCue(phrase="encoder builds bidirectional context representations while the decoder generates tokens autoregressively", cue_strength="strong"),
                MechanismCue(phrase="pre-norm transformers apply layer norm before the sublayer improving training stability", cue_strength="weak"),
                MechanismCue(phrase="cross-attention in decoder attends to encoder output allowing the decoder to condition on input context", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="transformers are just larger recurrent neural networks that process sequences", severity="major"),
                MisconceptionPhrase(phrase="the CLS token always summarises the entire input sequence", severity="minor"),
                MisconceptionPhrase(phrase="positional encodings are learned weights in the same way as embedding weights", severity="minor"),
                MisconceptionPhrase(phrase="layer norm and batch norm are interchangeable in transformer architectures", severity="major"),
                MisconceptionPhrase(phrase="deeper transformers always outperform shallower ones on all tasks", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="Describe the transformer architecture.", probe_type="definitional"),
                ProbeTemplate(template="Why do transformers need positional encodings and how do sinusoidal encodings provide position information?", probe_type="mechanism"),
                ProbeTemplate(template="What happens when you remove residual connections from a deep transformer?", probe_type="edge_case"),
                ProbeTemplate(template="Give a task where a pure encoder transformer would underperform a recurrent model.", probe_type="counterexample"),
                ProbeTemplate(template="How would you adapt the transformer architecture for a long-document summarisation task?", probe_type="application"),
            ],
        ),

        # ── S06: Loss Functions and Their Geometry ────────────────────────────
        SectionKB(
            section_id="S06",
            title="Loss Functions and Their Geometry",
            key_concepts=[
                "cross-entropy loss", "mean squared error", "hinge loss",
                "KL divergence", "log-likelihood", "calibration",
                "Huber loss", "margin",
            ],
            mechanism_cues=[
                MechanismCue(phrase="cross-entropy maximises the log-likelihood under the model's categorical distribution", cue_strength="strong"),
                MechanismCue(phrase="hinge loss enforces a margin by penalising scores within the margin linearly and ignoring scores beyond it", cue_strength="strong"),
                MechanismCue(phrase="KL divergence is asymmetric because P and Q play different roles as reference and approximating distributions", cue_strength="strong"),
                MechanismCue(phrase="MSE assumes a Gaussian noise model with constant variance making it sensitive to outliers", cue_strength="strong"),
                MechanismCue(phrase="Huber loss interpolates between MSE for small errors and MAE for large errors providing outlier robustness", cue_strength="strong"),
                MechanismCue(phrase="label smoothing regularises cross-entropy by spreading probability mass to non-target classes", cue_strength="weak"),
                MechanismCue(phrase="the gradient of cross-entropy with respect to pre-softmax logits is the difference between predicted and true distributions", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="mean squared error is always the best choice for regression because it is differentiable", severity="major"),
                MisconceptionPhrase(phrase="all loss functions converge to the same optimal model parameters", severity="major"),
                MisconceptionPhrase(phrase="cross-entropy and MSE are equivalent for classification tasks", severity="major"),
                MisconceptionPhrase(phrase="a lower loss always means a better model in terms of generalisation", severity="minor"),
                MisconceptionPhrase(phrase="KL divergence is a proper distance metric between distributions", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is cross-entropy loss?", probe_type="definitional"),
                ProbeTemplate(template="Why does cross-entropy loss lead to faster learning than MSE for classification?", probe_type="mechanism"),
                ProbeTemplate(template="When would MSE be a worse choice than Huber loss and what causes that?", probe_type="edge_case"),
                ProbeTemplate(template="Give a case where minimising cross-entropy produces a worse model than minimising Brier score.", probe_type="counterexample"),
                ProbeTemplate(template="How would you choose a loss function for a highly imbalanced multi-class classification task?", probe_type="application"),
            ],
        ),

        # ── S07: Batch Normalization ──────────────────────────────────────────
        SectionKB(
            section_id="S07",
            title="Batch Normalization",
            key_concepts=[
                "internal covariate shift", "mini-batch statistics",
                "running mean", "running variance",
                "learnable scale gamma", "learnable shift beta",
                "training vs inference mode", "layer normalisation",
            ],
            mechanism_cues=[
                MechanismCue(phrase="normalises pre-activation values using the mean and variance of the current mini-batch", cue_strength="strong"),
                MechanismCue(phrase="learnable affine parameters gamma and beta restore representational capacity after normalisation", cue_strength="strong"),
                MechanismCue(phrase="reduces internal covariate shift by stabilising the distribution of activations across layers", cue_strength="strong"),
                MechanismCue(phrase="at inference time uses exponential moving average of batch statistics rather than current batch", cue_strength="strong"),
                MechanismCue(phrase="batch norm allows higher learning rates by smoothing the loss landscape around parameter updates", cue_strength="weak"),
                MechanismCue(phrase="small batch sizes cause noisy statistics estimates making batch norm unstable", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="batch normalisation is equivalent to standard z-score normalisation applied to inputs", severity="major"),
                MisconceptionPhrase(phrase="batch normalisation is always applied after the activation function", severity="major"),
                MisconceptionPhrase(phrase="batch normalisation eliminates the need for a bias term in the preceding layer", severity="minor"),
                MisconceptionPhrase(phrase="layer norm and batch norm produce identical outputs for any batch size", severity="major"),
                MisconceptionPhrase(phrase="batch norm is only beneficial for very deep networks", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is batch normalisation?", probe_type="definitional"),
                ProbeTemplate(template="Why does batch norm use different statistics during training and inference, and what mechanism drives that?", probe_type="mechanism"),
                ProbeTemplate(template="What goes wrong with batch norm when batch size is 1?", probe_type="edge_case"),
                ProbeTemplate(template="Give a setting where layer norm outperforms batch norm and explain why.", probe_type="counterexample"),
                ProbeTemplate(template="How would you modify a pre-trained model that uses batch norm for fine-tuning on a small dataset?", probe_type="application"),
            ],
        ),

        # ── S08: Convolutional Neural Networks ───────────────────────────────
        SectionKB(
            section_id="S08",
            title="Convolutional Neural Networks",
            key_concepts=[
                "weight sharing", "local receptive field", "convolution kernel",
                "stride", "padding", "pooling", "translation equivariance",
                "feature map",
            ],
            mechanism_cues=[
                MechanismCue(phrase="weight sharing reduces parameters by applying the same filter at every spatial position", cue_strength="strong"),
                MechanismCue(phrase="local receptive fields exploit spatial locality so nearby pixels are more correlated than distant ones", cue_strength="strong"),
                MechanismCue(phrase="pooling provides translation invariance by aggregating activations within a neighbourhood", cue_strength="strong"),
                MechanismCue(phrase="deeper convolutional layers compose low-level edges into higher-level semantic features hierarchically", cue_strength="strong"),
                MechanismCue(phrase="convolution is equivariant to translation meaning shifting the input shifts the feature map by the same amount", cue_strength="strong"),
                MechanismCue(phrase="stride downsamples spatial resolution controlling the size of output feature maps", cue_strength="weak"),
                MechanismCue(phrase="dilated convolutions expand the receptive field without increasing parameter count or losing resolution", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="larger convolutional filters always capture more useful features", severity="major"),
                MisconceptionPhrase(phrase="CNNs can only be applied to image data", severity="major"),
                MisconceptionPhrase(phrase="pooling layers introduce translation invariance so convolution equivariance is irrelevant", severity="minor"),
                MisconceptionPhrase(phrase="increasing the number of filters always improves model accuracy", severity="minor"),
                MisconceptionPhrase(phrase="a fully-connected layer sees the full input so adding one replaces the need for convolutions", severity="major"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is weight sharing in CNNs?", probe_type="definitional"),
                ProbeTemplate(template="How does translation equivariance in convolutions arise from the weight sharing mechanism?", probe_type="mechanism"),
                ProbeTemplate(template="What happens to a CNN trained on fixed-size images when given a larger image at inference?", probe_type="edge_case"),
                ProbeTemplate(template="Give a task where a recurrent network would outperform a convolutional network and explain why.", probe_type="counterexample"),
                ProbeTemplate(template="How would you design a CNN for 1D audio classification to exploit temporal locality?", probe_type="application"),
            ],
        ),

        # ── S09: Reinforcement Learning Basics ───────────────────────────────
        SectionKB(
            section_id="S09",
            title="Reinforcement Learning Basics",
            key_concepts=[
                "Markov decision process", "value function", "policy",
                "Bellman equation", "temporal difference learning",
                "exploration-exploitation tradeoff", "policy gradient",
                "discount factor",
            ],
            mechanism_cues=[
                MechanismCue(phrase="Bellman equation decomposes the value of a state into the immediate reward plus the discounted value of the next state", cue_strength="strong"),
                MechanismCue(phrase="policy gradient theorem gives the gradient of expected return as the expected product of log-policy gradient and return", cue_strength="strong"),
                MechanismCue(phrase="temporal difference learning bootstraps value estimates from current estimates rather than waiting for episode completion", cue_strength="strong"),
                MechanismCue(phrase="epsilon-greedy balances exploration and exploitation by randomly acting with probability epsilon", cue_strength="strong"),
                MechanismCue(phrase="the discount factor gamma down-weights future rewards making the agent prefer earlier rewards", cue_strength="weak"),
                MechanismCue(phrase="advantage function measures how much better an action is compared to the average action in that state", cue_strength="weak"),
                MechanismCue(phrase="Q-function represents the expected return for taking action a in state s then following the policy", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="the RL agent just memorises the correct action for each state", severity="major"),
                MisconceptionPhrase(phrase="the discount factor gamma is only used to prevent infinite rewards in continuing tasks", severity="major"),
                MisconceptionPhrase(phrase="a higher reward signal always leads to faster learning", severity="minor"),
                MisconceptionPhrase(phrase="policy gradient methods are strictly better than value-based methods", severity="minor"),
                MisconceptionPhrase(phrase="exploration is only necessary at the start of training", severity="major"),
            ],
            probe_templates=[
                ProbeTemplate(template="What is a Markov decision process?", probe_type="definitional"),
                ProbeTemplate(template="How does the Bellman equation enable iterative value estimation without simulating full episodes?", probe_type="mechanism"),
                ProbeTemplate(template="What happens when the discount factor gamma approaches zero in a long-horizon task?", probe_type="edge_case"),
                ProbeTemplate(template="Give a case where a deterministic greedy policy outperforms epsilon-greedy despite less exploration.", probe_type="counterexample"),
                ProbeTemplate(template="How would you apply temporal difference learning to train a chess evaluation function?", probe_type="application"),
            ],
        ),

        # ── S10: Embeddings and Representation Learning ───────────────────────
        SectionKB(
            section_id="S10",
            title="Embeddings and Representation Learning",
            key_concepts=[
                "word embedding", "skip-gram", "negative sampling",
                "contrastive learning", "dimensionality reduction",
                "semantic similarity", "linear structure", "manifold hypothesis",
            ],
            mechanism_cues=[
                MechanismCue(phrase="word2vec skip-gram maximises the log-probability of context words given the centre word", cue_strength="strong"),
                MechanismCue(phrase="semantic relationships are encoded as linear vector offsets in the embedding space", cue_strength="strong"),
                MechanismCue(phrase="contrastive learning pulls representations of similar examples together and pushes dissimilar ones apart", cue_strength="strong"),
                MechanismCue(phrase="dimensionality reduction preserves relative pairwise distances in a lower-dimensional projection", cue_strength="strong"),
                MechanismCue(phrase="negative sampling approximates the full softmax normalisation by contrasting against randomly drawn negatives", cue_strength="strong"),
                MechanismCue(phrase="the manifold hypothesis posits that natural data lies on a low-dimensional manifold in the input space", cue_strength="weak"),
                MechanismCue(phrase="subword tokenisation with BPE embeddings handles morphologically rich languages and rare words", cue_strength="weak"),
            ],
            common_misconceptions=[
                MisconceptionPhrase(phrase="embeddings are just compressed one-hot encodings with reduced dimensionality", severity="major"),
                MisconceptionPhrase(phrase="words with similar meanings always have similar embeddings regardless of context", severity="major"),
                MisconceptionPhrase(phrase="larger embedding dimensions always produce better representations", severity="minor"),
                MisconceptionPhrase(phrase="contrastive learning requires labelled positive pairs to be effective", severity="major"),
                MisconceptionPhrase(phrase="PCA and t-SNE can both be used interchangeably for visualisation and downstream tasks", severity="minor"),
            ],
            probe_templates=[
                ProbeTemplate(template="What are word embeddings?", probe_type="definitional"),
                ProbeTemplate(template="How does the skip-gram objective cause semantically related words to end up with similar embedding vectors?", probe_type="mechanism"),
                ProbeTemplate(template="When would word embeddings give misleading similarity scores for homonyms?", probe_type="edge_case"),
                ProbeTemplate(template="Give a case where contextualised embeddings from a transformer outperform static word2vec embeddings.", probe_type="counterexample"),
                ProbeTemplate(template="How would you apply contrastive learning to build a sentence similarity model without explicit labels?", probe_type="application"),
            ],
        ),
    ]

    return {s.section_id: s for s in sections}


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import this in other modules
# ──────────────────────────────────────────────────────────────────────────────
KB: dict[str, SectionKB] = build_kb()
