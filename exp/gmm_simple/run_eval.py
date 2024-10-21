import argparse
from runexp.argparse import parse

if __name__ != "__main__":
    raise RuntimeError("this is a script")


# FIXME: this code is duplicated in ./eval.py and ./run_eval.py can we remove duplication without creating any import loop ?
def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("res-dir", type=str)
    parser.add_argument("--mz-min", type=float, default=778.0)
    parser.add_argument("--mz-max", type=float, default=786.0)
    parser.add_argument(
        "--override-dataset",
        type=str,
        default=None,
        help=(
            "this should be used to change the path of the dataset in case the "
            "evaluation is not run on the same device as the training. Pay "
            "attention to use the same version of the dataset."
        ),
    )
    parser.add_argument(
        "--fpr-iter", type=int, default=None, help="set to None (default) to not do any"
    )
    return parser


def main():
    # this will stop execution in the case of runexp arguments
    # --> allows to work even if cuda devices are not available here
    args = parse(get_parser())

    from eval import run_eval
    run_eval(args)


main()  # -> start with runexp
