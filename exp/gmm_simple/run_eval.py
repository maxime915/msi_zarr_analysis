from runexp.argparse import parse

if __name__ != "__main__":
    raise RuntimeError("this is a script")


def main():
    from eval import get_parser, run_eval
    run_eval(parse(get_parser()))


main()  # -> start with runexp
