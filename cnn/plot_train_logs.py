# %%
import pandas as pd
import matplotlib.pyplot as plt
from shared import PROJECT_DIR

# %%
df = pd.read_json(PROJECT_DIR / "logs/cnn_train.jsonl", lines=True)

# %%
plt.plot(df["epoch"], df["train_acc"], label="train")
plt.plot(df["epoch"], df["val_acc"], label="validation")
plt.plot(df["epoch"], df["test_acc"], label="test")
plt.legend()
plt.grid()
plt.savefig(PROJECT_DIR / "plots/cnn/training.png")
# %%
