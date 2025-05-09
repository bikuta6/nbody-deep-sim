import os
import subprocess
import pandas as pd
from gnn import GraphModel
from trainer import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Set up directories
os.makedirs("./data/train", exist_ok=True)
os.makedirs("./data/test", exist_ok=True)
os.makedirs("./gnn_weights", exist_ok=True)
os.makedirs("./results/gnn", exist_ok=True)

print("Directories created.")
# Generate data function

DEVICE = "cpu"

def generate_data(output_dir, num_files=10):
    for i in range(1, num_files + 1):
        random_seed = str(random.randint(0, 1000))
        output_file = os.path.join(output_dir, f"output_file_{i}.csv")
        cmd = " ".join(
            [
                "python",
                "./src/s01-dataset-generation.py",
                "--integrator",
                "leapfrog",
                "--n-bodies 3 25 50 100 250 500",
                "--output",
                output_file.replace("\\", "/"),
                "--steps",
                "1000",
                "--sim-type",
                "spiral",
                "--n-arms",
                "2",
                "--seed",
                random_seed,
                "--device",
                DEVICE,
            ]
        )
        subprocess.run(cmd, check=True, shell=True)


# Generate train and test data
if len(os.listdir("./data/train")) == 0:
    generate_data("./data/train", num_files=10)
if len(os.listdir("./data/test")) == 0:
    generate_data("./data/test", num_files=1)

print("Data generated.")

# Initialize model and trainer
model = GraphModel(
    input_dim=4,  # 3 (pos) + 1 (mass) + 3 (vel) + 3 * previous positions
    node_encoder_dims=None,
    encoder_dropout=0.0,
    gnn_dim=64,
    message_passing_steps=2,
    aggr="mean",
    output_hiddens=None,
    device=DEVICE,
    neighbors=10,
    scale_factor=1e6,
)

optimizer = Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.25, patience=5)
trainer = Trainer(
    model, optimizer=optimizer, device=DEVICE, dt=0.0001, scheduler=scheduler
)

print("Model and trainer initialized.")

# Train model

epoch_loss, _ = trainer.train_from_dir(
    epochs=100,
    batch_size=64,
    save_every=10,
    data_path="./data/train",
    save_path="./gnn_weights",
)


pd.DataFrame(epoch_loss, columns=["loss"]).to_csv(
    "./results/gnn/epoch_loss.csv", index=False
)

print("Training completed, evaluating model.")

# Test model
df_stepwise, df_rollout = trainer.test_from_dir(
    data_path="./data/test", stepwise=True, rollout=True, model_path="./gnn_weights"
)

print("Evaluation completed.")
# Save results to CSV
df_stepwise.to_csv("./results/gnn/test_results_stepwise.csv", index=True)
df_rollout.to_csv("./results/gnn/test_results_rollout.csv", index=True)

print("Training and testing completed. Results saved.")
