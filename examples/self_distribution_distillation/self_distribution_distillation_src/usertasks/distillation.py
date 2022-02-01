import os
import math
from types import MethodType

import torch

from fairseq import metrics, utils
from fairseq import options, checkpoint_utils
from fairseq.data import data_utils
from fairseq.data.data_utils import collate_tokens
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from self_distribution_distillation_src.usertasks.translation import TranslationUncertaintyTask

import logging
logger = logging.getLogger(__name__)


@register_task('distillation')
class DistillationTask(TranslationUncertaintyTask):
    def __init__(self, args, src_dict, tgt_dict, ensemble):
        super(DistillationTask, self).__init__(args, src_dict, tgt_dict)

        # Initialise ensemble to distil from
        self.teacher_ensemble = ensemble
        assert ensemble is not None

        # The top-k predictions to save from the ensemble
        self.teacher_ensemble_topk = args.teacher_ensemble_topk

        # Epsilon associated with topk calculation
        self.topk_eps = 1e-8

    @classmethod
    def add_args(cls, parser):
        """
        # Import all needed arguments for generation
        """
        TranslationUncertaintyTask.add_args(parser)

        # Specifies the distillation parameters
        parser.add_argument('--teacher-ensemble-path', type=str, help="Path to ensemble models to distill from")
        parser.add_argument('--teacher-ensemble-topk', type=int, default=-1, help="Save only top-k predictions")

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception("Could not infer language pair, please provide it explicitly")

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        models = None
        # Loading ensemble
        if cfg.teacher_ensemble_path is not None:

            logger.info("loading teacher ensemble from {}".format(cfg.teacher_ensemble_path))
            models, _ = checkpoint_utils.load_model_ensemble(
                cfg.teacher_ensemble_path.split(':'), task = TranslationTask.setup_task(cfg, **kwargs)
            )

            # Ensure models are on cuda
            use_cuda = torch.cuda.is_available() and not cfg.cpu

            # Optimize ensemble for generation (includes setting .eval())
            for model in models:

                # No need to store attentions
                model.make_generation_fast_(need_attn = False)

                # Move to cuda
                if use_cuda: model.cuda()

        return cls(cfg, src_dict, tgt_dict, ensemble = models)

    @torch.no_grad()
    def add_ensemble_logits(self, sample):

        # Compute a tensor of size (batch, len, models, vocab)
        sample['teacher_ensemble_logits'] = torch.stack(
            [model(**sample['net_input'])[0] for model in self.teacher_ensemble], dim = 2
        )

        # By default this is set to -1, meaning all predictions are saved
        if self.teacher_ensemble_topk > 0:

            # Lets only make use of the topk predictions
            topk = self.teacher_ensemble_topk

            # We also need the vocab size to determine smoothing
            vocab = sample['teacher_ensemble_logits'].size(-1)

            # These will be modified by only utilising the topk predictions
            predictions = sample['teacher_ensemble_logits']

            # We need the output to be in terms of log-probabilities
            predictions = torch.log_softmax(predictions, dim = -1)

            # Now extract the topk predictions
            topk_preds, topk_inds = torch.topk(predictions, k = topk, dim = -1)

            # Get the log-mass associated with these predictions
            topk_mass = torch.logsumexp(topk_preds, dim = -1, keepdims = True)
            topk_mass = torch.exp(topk_mass)

            # Get the log-mass associated with non-topk tokens
            botk_logmass = torch.log(1 - topk_mass + self.topk_eps) - math.log(vocab - topk)

            # Now create a new prediction
            newpred = torch.zeros_like(predictions) + botk_logmass
            newpred = newpred.scatter(-1, topk_inds, topk_preds)

            # Return this to the sample
            sample['teacher_ensemble_logits'] = newpred

        return sample

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        # Add teacher ensemble predictions to the sample
        sample = self.add_ensemble_logits(sample)

        # Now peform the training step according to the loss function/criterion
        return super(DistillationTask, self).train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad=ignore_grad
        )

    def valid_step(self, sample, model, criterion):

        # Add teacher ensemble predictions to the sample
        sample = self.add_ensemble_logits(sample)

        # Now peform the validation step according to the loss function/criterion
        return super(DistillationTask, self).valid_step(sample, model, criterion)


@register_task('distillationgauss')
class DistillationAndGaussTask(DistillationTask):
    def __init__(self, args, src_dict, tgt_dict, ensemble):
        super(DistillationAndGaussTask, self).__init__(args, src_dict, tgt_dict, ensemble)

    @classmethod
    def add_args(cls, parser):
        """
        # Import all needed arguments for generation
        """
        TranslationUncertaintyTask.add_args(parser)

        # Specifies the distillation parameters
        parser.add_argument('--teacher-ensemble-path', type=str, help="Path to ensemble models to distill from")

    @torch.no_grad()
    def add_ensemble_logits(self, sample):

        # Assume ensemble members produce gaussian predictions
        predictions = [model(**sample['net_input']) for model in self.teacher_ensemble]

        key = 'student_predictions_scale'
        # Compute a tensors of size (batch, len, models, vocab)
        sample['teacher_ensemble_means'] = torch.stack([pred[0] for pred in predictions], dim = 2)
        sample['teacher_ensemble_scales'] = torch.stack([pred[1][key] for pred in predictions], dim = 2)

        return sample