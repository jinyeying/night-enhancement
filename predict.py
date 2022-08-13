import os
import tempfile
import shutil
import argparse
from cog import BasePredictor, Path, Input
from PIL import Image
from dataset import ImageFolder

from ENHANCENET import ENHANCENET
from utils import *


class Predictor(BasePredictor):
    def setup(self):
        self.args = get_args()

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
    ) -> Path:
        exp_dir = "LOL"
        out_dir = "output"
        for d in [exp_dir, out_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        shutil.copy(str(image), os.path.join(exp_dir, "test.png"))

        gan = ENHANCENET(self.args)
        gan.build_model()
        gan.test()
        # result image will be saved it ./output
        out_image = Image.open(os.path.join(out_dir, os.listdir(out_dir)[0]))
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        out_image.save(str(out_path))
        return out_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="test", help="[train / test]")
    parser.add_argument("--dataset", type=str, default="LOL", help="dataset_name")
    parser.add_argument("--datasetpath", type=str, default="LOL", help="dataset_path")
    parser.add_argument(
        "--iteration",
        type=int,
        default=900000,
        help="The number of training iterations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The size of batch size"
    )
    parser.add_argument(
        "--print_freq", type=int, default=1000, help="The number of image print freq"
    )
    parser.add_argument(
        "--save_freq", type=int, default=100000, help="The number of model save freq"
    )
    parser.add_argument(
        "--decay_flag", type=str2bool, default=True, help="The decay_flag"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="The learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="The weight decay"
    )
    parser.add_argument(
        "--atten_weight", type=int, default=0.5, help="Weight for Attention Loss"
    )
    parser.add_argument(
        "--use_gray_feat_loss",
        type=str2bool,
        default=True,
        help="use Structure and HF-Features Consistency Losses",
    )
    parser.add_argument(
        "--feat_weight",
        type=int,
        default=1,
        help="Weight for Structure and HF-Features Consistency Losses",
    )
    parser.add_argument("--adv_weight", type=int, default=1, help="Weight for GAN Loss")
    parser.add_argument(
        "--identity_weight", type=int, default=5, help="Weight for Identity Loss"
    )
    parser.add_argument(
        "--ch", type=int, default=64, help="base channel number per layer"
    )
    parser.add_argument("--n_res", type=int, default=4, help="The number of resblock")
    parser.add_argument(
        "--n_dis", type=int, default=6, help="The number of discriminator layer"
    )
    parser.add_argument(
        "--img_size", type=int, default=512, help="The training size of image"
    )
    parser.add_argument(
        "--img_ch", type=int, default=3, help="The size of image channel"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory name to save the results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Set gpu mode; [cpu, cuda]",
    )
    parser.add_argument("--benchmark_flag", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument(
        "--im_suf_A",
        type=str,
        default=".png",
        help="The suffix of test images [.png / .jpg]",
    )
    return parser.parse_args([])
