from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 400
    shots: tuple = (1, 3, 5)
    max_shot: int = 5
    train_episodes: int = 1200
    val_episodes: int = 150
    test_episodes: int = 1200
    ktp_angles: int = 12
    ktp_lengths: tuple = (7, 11, 15)
    ktp_width: int = 1
    ktp_tau: float = 0.1
    assouad_small_scales: tuple = (1, 2, 4)
    assouad_large_scales: tuple = (16, 32, 64)
    assouad_tau: float = 0.1
    assouad_quantile: float = 0.95
    assouad_band_px: int = 8
    feature_dim: int = 256
    lambda_assouad: float = 0.5
    beta_assouad: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    seed: int = 42
    threshold: float = 0.5
    train_fraction: float = 0.8
    val_fraction: float = 0.1

DATASET_EPISODES = {
    "kvasir": {"train": 1200, "val": 150, "test": 300},
    "isic2016": {"train": 1200, "val": 150, "test": 1200},
}
