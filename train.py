import torch

def main(args):
    torch.set_num_threads(args.num_threads)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=16)
    args = parser.parse_args()

    main(args)
