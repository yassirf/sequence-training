import logging
from dataclasses import dataclass, field
import torch
from omegaconf import II

from fairseq.data.data_utils import collate_tokens
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from self_distribution_distillation_src.uncertainty.categorical import (
    compute_token_uncertainties, compute_sequence_uncertainties)
from self_distribution_distillation_src.uncertainty.dirichlet import (
    compute_token_dirichlet_uncertainties, compute_sequence_dirichlet_uncertainties)
from self_distribution_distillation_src.uncertainty.gaussian import (
    compute_token_gaussian_uncertainties, compute_sequence_gaussian_uncertainties,
    compute_token_gaussian_dirichlet_uncertainties, compute_sequence_gaussian_dirichlet_uncertainties)


@dataclass
class TranslationUncertaintyConfig(TranslationConfig):
    compute_uncertainty: int = field(
        default=0,
        metadata={"help": "Whether or not to compute uncertainty"},
    )
    uncertainty_class: str = field(
        default='categorical',
        metadata={"help": "Nature of model output"}
    )
    ood_num_samples: int = field(
        default=25,
        metadata={"help": "Number of samples to draw to compute uncertainties"}
    )
    ood_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scaling of categorical ensemble"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_uncertainty", dataclass=TranslationUncertaintyConfig)
class TranslationUncertaintyTask(TranslationTask):

    cfg: TranslationUncertaintyConfig

    def __init__(self, args, src_dict, tgt_dict):
        super(TranslationUncertaintyTask, self).__init__(args, src_dict, tgt_dict)
        logger = logging.getLogger("fairseq_cli.task")
        logger.info("initialising the uncertainty translation module")

        self.args = args

        # Set default token and sentence uncertainty estimators
        self.compute_token_uncertainties = compute_token_uncertainties
        self.compute_sequence_uncertainties = compute_sequence_uncertainties
        if args.uncertainty_class.startswith("dirichlet"):
            self.compute_token_uncertainties = compute_token_dirichlet_uncertainties
            self.compute_sequence_uncertainties = compute_sequence_dirichlet_uncertainties
        elif args.uncertainty_class.startswith("gaussian"):
            self.compute_token_uncertainties = compute_token_gaussian_uncertainties
            self.compute_sequence_uncertainties = compute_sequence_gaussian_uncertainties
        elif args.uncertainty_class.startswith("dirgaussian"):
            self.compute_token_uncertainties = compute_token_gaussian_dirichlet_uncertainties
            self.compute_sequence_uncertainties = compute_sequence_gaussian_dirichlet_uncertainties

    @classmethod
    def add_args(cls, parser):
        """
        # Import all needed arguments for generation
        """
        TranslationTask.add_args(parser)

        # Specifies the uncertainty estimation parameters
        parser.add_argument('--compute_uncertainty', type=int, default=0,
                            help="Whether or not to compute uncertainty")
        parser.add_argument('--uncertainty_class', type=str, default='categorical',
                            choices=['categorical', 'dirichlet', 'gaussian', 'dirgaussian'], help="Type of model output")
        parser.add_argument('--ood_num_samples', type=int, default=25,
                            help="Number of samples to draw to compute uncertainties")
        parser.add_argument('--ood_temperature', type=float, default=1.0,
                            help="Temperature scaling of categorical ensemble")

    @torch.no_grad()
    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):

        # Get all predictions
        hypos = generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)

        if self.args.compute_uncertainty:
            # Compute token and sequence level uncertainties
            self.add_uncertainties(sample, hypos, models)

        return hypos

    def add_uncertainties(self, sample, hypos, models):

        # Get the number of predictions
        nbest = min(len(sent) for sent in hypos)

        # Converts a list of 1d tensors into a padded 2d tensor
        tokens = collate_tokens([out['tokens'] for sent in hypos for out in sent[:nbest]],
                                eos_idx=self.tgt_dict.eos(), pad_idx=self.tgt_dict.pad())

        # Shifts the output sentences by a single step
        prev_output = collate_tokens([out['tokens'] for sent in hypos for out in sent[:nbest]],
                                     eos_idx=self.tgt_dict.eos(), pad_idx=self.tgt_dict.pad(),
                                     move_eos_to_beginning=True)

        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        prev_tokens = sample['net_input']['prev_output_tokens']

        sample['net_input']['src_tokens'] = torch.repeat_interleave(sample['net_input']['src_tokens'], nbest, dim=0)
        sample['net_input']['src_lengths'] = torch.repeat_interleave(sample['net_input']['src_lengths'], nbest, dim=0)
        sample['net_input']['prev_output_tokens'] = prev_output

        # Get all net_outputs and remove the extra information
        net_outputs = [model(**sample['net_input']) for model in models]
        net_extra   = [extra for (z, extra) in net_outputs]
        net_outputs = [z for (z, extra) in net_outputs]

        # Restate the original inputs
        sample['net_input']['src_tokens'] = src_tokens
        sample['net_input']['src_lengths'] = src_lengths
        sample['net_input']['prev_output_tokens'] = prev_tokens

        # Get the padding mask
        mask = tokens.eq(self.tgt_dict.pad())

        # Get sequence lengths
        num_of_tokens = torch.sum(~mask, dim=1)

        # Get token level uncertainties
        entropy_expected, expected_entropy, mutual_information = self.compute_token_uncertainties(
            self.args, net_outputs, net_extra
        )

        # Get sentence level uncertainties
        scores, log_probs, token_log_probs = self.compute_sequence_uncertainties(
            self.args, net_outputs, net_extra, tokens, num_of_tokens, mask
        )

        # Mask out any uncertainty corresponding to padding
        if mask.any():
            entropy_expected.masked_fill_(mask, 0.0)
            expected_entropy.masked_fill_(mask, 0.0)
            mutual_information.masked_fill_(mask, 0.0)

        # Store all results in hypothesis
        for i, sent in enumerate(hypos):
            for j, hypo in enumerate(sent[:nbest]):
                ind = i * nbest + j

                hypo['token_uncertainties'] = {
                    'entropy_expected': entropy_expected[ind],
                    'expected_entropy': expected_entropy[ind],
                    'mutual_information': mutual_information[ind],
                    'token_pe_TU': -token_log_probs[ind],
                }

                hypo['sequence_uncertainties'] = {
                    'pe_entropy_expected': entropy_expected[ind].sum() / num_of_tokens[ind],
                    'expected_entropy': expected_entropy[ind].sum() / num_of_tokens[ind],
                    'pe_mutual_information': mutual_information[ind].sum() / num_of_tokens[ind],
                    'pe_sTU': scores[ind],
                    'log-prob': log_probs[ind],
                }