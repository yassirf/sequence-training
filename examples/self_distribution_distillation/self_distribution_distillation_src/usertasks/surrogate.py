import os
import logging
from dataclasses import dataclass, field
from wsgiref.simple_server import demo_app
import torch
from omegaconf import II

from fairseq import utils, checkpoint_utils
from fairseq.data import data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from .translation import TranslationUncertaintyConfig, TranslationUncertaintyTask

logger = logging.getLogger(__name__)


@dataclass
class TranslationSurrogateUncertaintyConfig(TranslationUncertaintyConfig):
    surrogate_uncertainty_path: str = field(
        default='',
        metadata={"help": "Path to ensemble models to distill from"},
    )
    compute_uncertainty: int = field(
        default=0,
        metadata={"help": "Whether or not to compute uncertainty"},
    )
    uncertainty_class: str = field(
        default='categorical',
        metadata={"help": "Nature of model output"},
    )
    ood_num_samples: int = field(
        default=25,
        metadata={"help": "Number of samples to draw to compute uncertainties"},
    )
    ood_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scaling of categorical ensemble"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_surrogate_uncertainty", dataclass=TranslationSurrogateUncertaintyConfig)
class TranslationSurrogateUncertaintyTask(TranslationUncertaintyTask):

    cfg: TranslationSurrogateUncertaintyConfig

    def __init__(self, args, src_dict, tgt_dict, surrogate):
        super(TranslationSurrogateUncertaintyTask, self).__init__(args, src_dict, tgt_dict)
        
        # Set surrogate models
        self.surrogate = surrogate
        assert surrogate is not None

    @classmethod
    def add_args(cls, parser):
        """
        # Import all needed arguments for generation
        """
        TranslationTask.add_args(parser)

        # Specifies the uncertainty estimation parameters
        parser.add_argument('--surrogate-uncertainty-path', type=str, required=True,
                            help="Path to ensemble models to distill from")
        parser.add_argument('--compute_uncertainty', type=int, default=0,
                            help="Whether or not to compute uncertainty")
        parser.add_argument('--uncertainty_class', type=str, default='categorical',
                            choices=['categorical', 'dirichlet', 'gaussian', 'laplace'], help="Type of model output")
        parser.add_argument('--ood_num_samples', type=int, default=25,
                            help="Number of samples to draw to compute uncertainties")
        parser.add_argument('--ood_temperature', type=float, default=1.0,
                            help="Temperature scaling of categorical ensemble")

    @classmethod
    def setup_task(cls, cfg: TranslationSurrogateUncertaintyConfig, **kwargs):
        """
        Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception("Could not infer language pair, please provide it explicitly")

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        surrogate = None
        # load surrogate models for uncertainty estimation
        if cfg.surrogate_uncertainty_path is not None:

            logger.info("loading surrogate models from {}".format(cfg.surrogate_uncertainty_path))
            surrogate, _ = checkpoint_utils.load_model_ensemble(
                cfg.teacher_ensemble_path.split(':'), task = TranslationSurrogateUncertaintyTask.setup_task(cfg, **kwargs)
            )

            # Ensure models are on cuda
            use_cuda = torch.cuda.is_available() and not cfg.cpu

            # Optimize ensemble for generation (includes setting .eval())
            for model in surrogate:

                # No need to store attentions
                model.make_generation_fast_(need_attn = False)

                # Move to cuda
                if use_cuda: model.cuda()

        return cls(cfg, src_dict, tgt_dict, surrogate)

    @torch.no_grad()
    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):

        # Get all predictions
        hypos = generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)

        if self.args.compute_uncertainty:
            # Compute token and sequence level uncertainties
            self.add_uncertainties(sample, hypos, self.surrogate)

        return hypos