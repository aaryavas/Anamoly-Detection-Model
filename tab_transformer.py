import torch
import numpy as np
from pathlib import Path

from tabtransformers.tabular_datamodule_v2 import TabularDataModuleForClassification
from tabtransformers.tabular_module_v2 import TabularModuleForClassification
from tabtransformers.train import CLASSIFICATION_FEATURES

# Replace these imports with the actual modules where TabularDataModuleForClassification
# and TabularModuleForClassification are defined.
# For example:
# from my_data_module import TabularDataModuleForClassification
# from my_tabular_module import TabularModuleForClassification

# 1. Load your pre-trained model checkpoint
checkpoint_path = "../TabTransformer-Classification-Full/v3d0gdh2/checkpoints/epoch=1-step=767.ckpt"  # placeholder path
model = TabularModuleForClassification.load_from_checkpoint(checkpoint_path).to("cpu")
model.return_attention = True
model.eval()  # set to etrainuation mode

# 2. Prepare the data module as you typically would
data_module = TabularDataModuleForClassification(
    data_dir=Path("../../law/tmp_data"),  # placeholder path
    categorical_columns=CLASSIFICATION_FEATURES,  # placeholder columns
    batch_size=1,
    num_workers=4,
    compute_attorney_specialization=False,
)

# 3. Set up the data module for training (or any split you want)
data_module.setup()
import pickle

train_dataset = data_module.train_dataset
df = data_module.train_df

# 4. Collect attention outputs from your model
attention_outputs_dict = {}
with torch.no_grad():
    for idx in range(len(train_dataset)):
        batch = train_dataset[idx]
        motion_id = df.iloc[idx]["MotionID"]
        # model(...) returns (predictions, attention) when return_attention=True
        predictions, attention = model(
            batch["categorical"].unsqueeze(0), batch["numerical"].unsqueeze(0)
        )
        # Convert attention to CPU and NumPy for easy saving
        attention_outputs_dict[motion_id] = attention.cpu().numpy().flatten()

# 5. Save attention to file as a pickle
with open("attention_outputs_train.pkl", "wb") as f:
    pickle.dump(attention_outputs_dict, f)
print("Attention outputs saved to attention_outputs_train.pkl")