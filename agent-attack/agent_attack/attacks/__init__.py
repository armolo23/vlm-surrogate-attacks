from .bim import bim
from .clip_attack import clip_attack
from .clip_attack_background import clip_attack_background
from .pgd import pgd


def get_attack_fn(attack):
    if attack == "pgd":
        return pgd
    elif attack == "bim":
        return bim
    elif attack == "clip_attack":
        return clip_attack
    elif attack == "clip_attack_background":
        return clip_attack_background
    else:
        raise ValueError(f"Attack {attack} not supported.")
