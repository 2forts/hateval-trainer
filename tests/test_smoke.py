from pathlib import Path
from hateval_trainer.config import Config
from hateval_trainer.data import load_dataset, clean_text, balance_classes

# This is a minimal smoke test that only checks the data pipeline up to balancing.

def test_data_pipeline_smoke(tmp_path: Path):
  # Create a tiny CSV with the expected schema
  csv = tmp_path / "tiny.csv"
  csv.write_text("id;text;HS\n1;hola mundo;0\n2;mensaje malo;1\n3;otro texto;0\n", encoding="utf-8")
  
  df = load_dataset(csv)
  df = clean_text(df)
  df = balance_classes(df)
  
  assert set(df.columns) == {"mensaje", "intensidad"}
  assert len(df) >= 3
