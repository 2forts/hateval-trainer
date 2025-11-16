from pathlib import Path
from hateval_trainer.config import Config
from hateval_trainer.data import load_dataset   # depende de cómo cargues tus datos
from hateval_trainer.train import cross_validate

# 1) Configuración con FLAG de ablation activado
cfg = Config(
    csv_path=Path("data/train.csv"),
    output_model=Path("outputs/model_ablation"),
    cv_use_dense_ablation=True,   # <<--- activar ablación
    cv_use_hybrid=False,          # <<--- desactivar híbrido
    k=5,                           # si usas CV a 5 folds
    epochs=10,
    hybrid_epochs=5
)

# 2) Cargar datos
df = load_dataset(cfg.csv_path)

# 3) Ejecutar CV
results = cross_validate(df, cfg, k=5)

print(results)