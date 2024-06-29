import torch

def main(args):
    torch.set_num_threads(args.num_threads)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    ckpt = torch.load(args.model_path)

    dataset = ckpt['dataset']
    assert dataset == "tpu_tile"

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--min_num_steps_n", type=int, default=None)
    parser.add_argument("--min_num_steps_e", type=int, default=None)
    parser.add_argument("--max_num_steps_n", type=int, default=None)
    parser.add_argument("--max_num_steps_e", type=int, default=None)
    args = parser.parse_args()

    main(args)
