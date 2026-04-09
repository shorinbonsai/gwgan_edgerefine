from dataclasses import dataclass, field
from typing import List, Dict

# --------------------------
# Configuration
# --------------------------
@dataclass
class Config:
    seed:int = 42
    dataset_name: str =  'PROTEINS' #PROTEINS, #ENZYMES, MUTAG
    batch_size: int = 64
    train_split: float = 0.7
    val_split: float = 0.15

    noise_dim: int = 16
    class_embed_dim: int = 8
    hidden_dim_gen: int = 32
    hidden_dim_dis: int = 32

    n_critic: int = 5
    lambda_gp: float = 10.0
    epochs: int = 50
    patience: int = 12
    lr_gen: float = 2e-4
    lr_dis: float = 5e-4
    betas_gen: tuple = (0.5, 0.9)
    betas_dis: tuple = (0.5, 0.9)

    start_temperature: float = 2.0
    end_temperature: float = 0.5

    num_samples_per_label: int = 150
    samples_per_batch: int = 50
    num_bins: int = 10
    vis_num_samples: int = 6

    # --------------------------
    # In-Training GA Refinement
    # --------------------------
    # Master toggle: when True, the Generator runs the GA inside forward()
    # during training epochs (after warmup). When False, the Generator
    # uses plain top-k edges with no GA step.
    use_refinement: bool = True

    # Number of warmup epochs to train the Generator without GA refinement.
    refinement_warmup_epochs: int = 3
    # Run GA every N epochs after warmup.
    refinement_interval: int = 1 

    # Training-time GA parameters (kept lighter than a post-hoc pass
    # since the GA runs inside every generator forward call)
    training_refiner_pop: int = 100
    training_refiner_gens: int = 50

    # GA operator and reproduction settings
    crossover_probability: float = 0.5
    mutation_probability: float = 0.8

    # Weights for the 9 graph operations:
    # [Toggle, Hop, Add, Delete, Swap, LocalToggle, LocalAdd, LocalDelete, Null]
    refinement_op_weights: List[float] = field(default_factory=lambda: [
        0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0
    ])

    # Edge count penalty weight for the GA fitness function
    edge_penalty_weight: float = 0.01

    # MMD kernel bandwidths (used by GA fitness and evaluation)
    gammas: Dict[str, float] = field(default_factory=lambda: {'degree': 1.0, 'clustering': 1.0, 'spectral': 0.1})

    # Component weights for the combined MMD score (used by GA fitness and evaluation)
    weights: Dict[str, float] = field(default_factory=lambda: {'degree': 0.3, 'clustering': 0.4, 'spectral': 0.3})

    # How often to save GA logs during training (0 = never)
    # When > 0, logs are saved for the first graph of the first batch
    # every N epochs. Useful for debugging GA convergence.
    refinement_log_interval: int = 10

    save_dir: str = './saved_models/'
    results_dir: str = './results/'
    data_dir: str = './data/'