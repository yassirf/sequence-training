import logging
from dataclasses import dataclass, field
import torch
from omegaconf import II

from fairseq.tasks import register_task

from self_distribution_distillation_src.usertasks.translation import (
    TranslationUncertaintyConfig, TranslationUncertaintyTask)


@dataclass
class TranslationMIMOConfig(TranslationUncertaintyConfig):
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
    batch_repetition: float = field(
        default=1,
        metadata={"help": "Number of shuffled batch repetitions"}
    )
    input_repetition: float = field(
        default=0.0,
        metadata={"help": "Fraction of additional repeated inputs"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_mimo", dataclass=TranslationMIMOConfig)
class TranslationMIMOTask(TranslationUncertaintyTask):

    cfg: TranslationMIMOConfig

    def __init__(self, args, src_dict, tgt_dict):
        super(TranslationMIMOTask, self).__init__(args, src_dict, tgt_dict)
        logger = logging.getLogger("fairseq_cli.task")
        logger.info("initialising the mimo translation module")

    @classmethod
    def add_args(cls, parser):
        """
        # Import all needed arguments for generation
        """
        TranslationUncertaintyTask.add_args(parser)

        # Specifies the mimo training parameters
        parser.add_argument('--batch_repetition', type=int, default=1,
                            help="Number of shuffled batch repetitions")
        parser.add_argument('--input_repetition', type=float, default=0.0,
                            help="Fraction of additional repeated inputs")

    def add_input_repetition(self, sample):
        """
        Input:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        # Number of repetitions in batch
        repetitions = self.args.input_repetition

        # Get the number of examples in batch
        nsamples = sample['id'].size(0)

        # Special case if no added samples
        if abs(repetitions) < 1e-6: return sample

        # Create repeated samples (for two headed model) and ensure is divisible by batch multiple
        repsamples = int(repetitions * nsamples)

        # Ensure batch is divisible by multiple of two for base mimo
        multip = 2

        # Add additional parameters to ensure divisibility
        repsamples += multip - repsamples % multip if repsamples % multip else 0

        # Now create the augmented sample
        sample['ntokens'] *= (1 + repetitions)

        # Create permutation long tensor
        p = torch.stack([torch.arange(repsamples), torch.arange(repsamples)], dim=1).reshape(-1)
        p = torch.cat([p, torch.arange(repsamples, nsamples)])

        # Meta and target information permuted
        for key, value in sample.items():
            if isinstance(value, torch.LongTensor):
                sample[key] = value[p]

        # Input information augmented
        for key, value in sample['net_input'].items():
            if isinstance(value, torch.LongTensor):
                sample['net_input'][key] = value[p]

        return sample

    def add_batch_repetition(self, sample):
        """
        Input:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """

        # Number of repetitions in batch
        repetitions = round(self.args.batch_repetition)

        # For single repetition return the same sample
        if repetitions == 1: return sample

        # Get the number of examples in batch
        nsamples = sample['id'].size(0)

        # Create a permutation based on this
        p = torch.cat([torch.randperm(nsamples) for _ in range(repetitions - 1)])

        # Now create the augmented sample
        sample['ntokens'] *= repetitions

        # Meta and target information permuted
        for key, value in sample.items():
            if isinstance(value, torch.LongTensor):
                sample[key] = torch.cat([value, value[p]])

        # Input information augmented
        for key, value in sample['net_input'].items():
            if isinstance(value, torch.LongTensor):
                sample['net_input'][key] = torch.cat([value, value[p]])

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

        # Modify the sample with batch and input repetition
        sample = self.add_input_repetition(sample)
        sample = self.add_batch_repetition(sample)

        return super(TranslationMIMOTask, self).train_step(
            sample = sample,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            update_num = update_num,
            ignore_grad = ignore_grad
        )