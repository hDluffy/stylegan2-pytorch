import argparse

import torch
from torchvision import utils
from model_script import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        #g_ema.eval().half()
        ###配合e4e输入(1,18,512)###
        #sample_z_ = torch.randn(args.sample, args.latent, device=device)
        sample_z_ = torch.randn(args.sample, 18, args.latent, device=device)
        #sample_z_ = torch.randn(args.sample, 18, args.latent, device=device).half()
        traced_script_module_generator = torch.jit.trace(g_ema, (sample_z_))
        traced_script_module_generator.save('generator.pt')
        for i in tqdm(range(args.pics)):
            ###配合e4e输入(1,18,512)###
            #sample_z = torch.randn(args.sample, args.latent, device=device)
            sample_z = torch.randn(args.sample, 18, args.latent, device=device)
            #sample_z = torch.randn(args.sample, 18, args.latent, device=device).half()

            #sample = g_ema(
            #    sample_z, truncation=args.truncation, truncation_latent=mean_latent
            #)
            sample = traced_script_module_generator(sample_z)

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
