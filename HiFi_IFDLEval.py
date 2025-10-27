import argparse
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

from DataUtils import FakePartsV2DatasetBase, collate_skip_none, standardise_predictions
from HiFi_Net_loc import config as hifi_config
from utils.utils import restore_weight_helper

# Log config for streaming outputs to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="HiFi-IFDL Inference on FakePartsV2")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--results', type=str, required=True, help="Directory to save results")
    parser.add_argument('--data_csv', type=str, default=None, help="Path to the dataset index CSV")
    parser.add_argument('--done_csv_list', nargs='+', default=[], help="List of CSV files with done samples")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for inference")
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'dm'])

    return parser.parse_args()


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def append_to_csv(df: pd.DataFrame, csv_path: Path):
    """Append a DataFrame to a CSV file, creating the file with a header if it doesn't exist."""
    is_new_file = not csv_path.exists()
    df.to_csv(csv_path, mode='a', header=is_new_file, index=False)


def main():
    args = parse_args()

    # Load model
    log.info("Loading HiFi-IFDL model...")
    hifi_args, _, FENet, SegNet, FENet_dir, SegNet_dir = hifi_config(args)
    if hifi_args.loss_type == 'ce':
        FENet = restore_weight_helper(FENet, "weights/HRNet", 225000)
        SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 225000)
    elif hifi_args.loss_type == 'dm':
        FENet = restore_weight_helper(FENet, "weights/HRNet", 315000)
        SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 315000)

    FENet.eval()
    SegNet.eval()

    # Prepare dataset
    log.info("Preparing dataset...")
    dataset = FakePartsV2Dataset(
        data_root=args.data_root,
        mode="frame",
        csv_path=args.data_csv,
        done_csv_list=args.done_csv_list,
        model_name="HiFi_IFDL",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                            collate_fn=collate_skip_none)

    results_path = Path(args.results)
    results_path.mkdir(parents=True, exist_ok=True)
    output_csv = results_path / "predictions.csv"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FENet.to(device)
    SegNet.to(device)

    log.info("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            if batch is None:
                continue
            images, labels, metas = batch
            images = images.to(device)

            output = FENet(images)
            _, mask_binary, _, _, _, _ = SegNet(output, images)

            scores = torch.amax(mask_binary, dim=(-2, -1)).squeeze().cpu().numpy()
            preds = (scores > 0.5).astype(int)

            metas["score"] = scores
            metas["pred"] = preds

            df = standardise_predictions(metas)
            append_to_csv(df, output_csv)


if __name__ == "__main__":
    main()
