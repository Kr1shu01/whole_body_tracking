import wandb

REGISTRY_NAME = "motions"
COLLECTION_NAME = "Vdance3"

run = wandb.init(project="csv_to_npz", name=COLLECTION_NAME)

logged_artifact = run.log_artifact(artifact_or_path="./motion/Vdance3/motion.npz", name=COLLECTION_NAME, type=REGISTRY_NAME)

run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
