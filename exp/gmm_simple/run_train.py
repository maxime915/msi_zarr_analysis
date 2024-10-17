from runexp.config_file import runexp_main
from config import PSConfig

if __name__ != "__main__":
    raise RuntimeError("this is a script")


@runexp_main
def peak_sense_train(config: PSConfig):
    from train import train
    train(config)
