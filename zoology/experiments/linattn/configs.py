import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

factory_kwargs = {
        "num_kv_pairs": 4,
    }

models = []
for causal_bool in [True, False]:
    models.append(ModelConfig(
        d_model=128,
        n_layers=2,
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.linattn.MHLA",
            kwargs={"dropout": 0.1, "num_heads": 1, "causal": causal_bool},
        ),
        name="LinAttn",
    ))

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        config = TrainConfig(
            data=DataConfig(
                cache_dir="/scr/ayz/zoology_cache",
                train_configs=[MQARConfig(num_examples=10_000, vocab_size=256, input_seq_len=64, **factory_kwargs)],
                test_configs=[MQARConfig(num_examples=1_000, vocab_size=256, input_seq_len=64, **factory_kwargs)],
            ),
            learning_rate=lr,
            model=model,
            logger=LoggerConfig(
                project_name="zoology",
                entity="geometric-meta-learning"
            ),
            run_id=f"{model.name}-lr{lr:.1e}",
        )
        configs.append(config)