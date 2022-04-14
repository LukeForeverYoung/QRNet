from .trans_vg import TransVG, TransVGSwin


def build_model(args):
    return TransVGSwin(args)
