from .owl import OWL
from .dlp import DLP
from .blk import BLK
from .er import ER
from .atp import ATP
from .alphapruning import AlphaPruning
from .lod import lod_cal


class Uniform:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.A = 5

    def get_layer_sp(self, args):
        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = self.model.model.layers
        elif "opt" in args.base_model:
            layers = self.model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = self.model.transformer.h
        else:
            layers = self.model.model.layers
        return [args.final_s] * len(layers)
