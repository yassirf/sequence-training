import math
from dataclasses import dataclass, field

import torch
from torch.distributions.normal import Normal
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion
)
from fairseq.dataclass import FairseqDataclass
from .distillation import KLDivergenceCriterionConfig, KLDivergenceCriterion
from self_distribution_distillation_src.usertasks.distillation import DistillationAndGaussTask
from omegaconf import II


@dataclass
class GaussKLDivergenceAndGaussCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    self_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio of default to self loss"}
    )
    temperature_scale_est: float = field(
        default=1.0,
        metadata={"help": "temperature scaling teacher predictions for dirichlet estimation"}
    )
    temperature_scale_num: float = field(
        default=1.0,
        metadata={"help": "temperature scaling alphas in kl divergence for numerical stability"}
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def gaussian_kl_divergence(
        gaussian_mean,
        gaussian_scale,
        teacher_gaussian_mean,
        teacher_gaussian_scale,
        ignore_mask = None,
        reduce = True
):

    # Get student gaussian
    gaussian = Normal(gaussian_mean.unsqueeze(dim = 2), gaussian_scale.unsqueeze(dim = 2))

    # Get teacher gaussian
    teacher_gaussian = Normal(teacher_gaussian_mean, teacher_gaussian_scale)

    # Get the non-summed loss (batch, len, models, vocab)
    loss = torch.distributions.kl.kl_divergence(teacher_gaussian, gaussian)

    # Sum over vocabulary and average over models (batch, len)
    loss = loss.sum(dim = -1).mean(dim = 2)

    # Mask out padding elements
    if ignore_mask is not None: loss.masked_fill_(ignore_mask, 0.0)

    if reduce: loss = loss.sum()
    return loss


@register_criterion("gauss_kl_divergence_distillation_and_gauss", dataclass=GaussKLDivergenceAndGaussCriterionConfig)
class GaussKLDivergenceAndGaussCriterion(KLDivergenceCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            self_ratio=0.0,
            temperature_scale_num=1.0,
    ):
        super(GaussKLDivergenceAndGaussCriterion, self).__init__(
            task = task,
            sentence_avg = sentence_avg,
            label_smoothing = label_smoothing,
            ignore_prefix_size = ignore_prefix_size,
            report_accuracy = report_accuracy,
            temperature_scale_num = temperature_scale_num,
            temperature_scale_est=1.0,
        )

        # This loss is only compatible with DistillationAndGaussTask
        assert isinstance(task, DistillationAndGaussTask)

        # For mixing losses
        self.self_ratio = self_ratio

    @classmethod
    def build_criterion(cls, cfg: GaussKLDivergenceAndGaussCriterionConfig, task):
        return GaussKLDivergenceAndGaussCriterion(
            task = task,
            sentence_avg = cfg.sentence_avg,
            label_smoothing = cfg.label_smoothing,
            ignore_prefix_size = cfg.ignore_prefix_size,
            report_accuracy = cfg.report_accuracy,
            self_ratio = cfg.self_ratio,
            temperature_scale_num = cfg.temperature_scale_num,
        )

    def compute_gauss_loss(self, model, net_output, sample, reduce=True):
        """
        Estimate and compute the kl-divergence between student and teacher dirichlets
        """

        # Get teacher predictions
        teacher_gaussian_mean = sample['teacher_ensemble_means']
        teacher_gaussian_scale = sample['teacher_ensemble_scales']

        # Get student predictions
        gaussian_mean = net_output[0]
        gaussian_scale = net_output[1]['student_predictions_scale']

        # Compute loss
        loss = gaussian_kl_divergence(
            gaussian_mean = gaussian_mean,
            gaussian_scale = gaussian_scale,
            teacher_gaussian_mean = teacher_gaussian_mean,
            teacher_gaussian_scale = teacher_gaussian_scale,
            ignore_mask = self.get_padding_mask(sample),
            reduce = reduce
        )

        return loss

    def forward(self, model, sample, reduce=True):
        """
        Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # Get prediction
        net_output = model(**sample["net_input"])

        # Get tracking metrics (no grad)
        ls_loss, nll_loss = self.compute_nll_loss(model, net_output, sample, reduce)

        # Zero element
        zero = torch.zeros_like(ls_loss)

        # Get kl-divergence loss only during training
        kl_loss = self.compute_kl_loss(model, net_output, sample, reduce) if model.training else zero

        # Get dirichlet loss only during training
        gauss_loss = self.compute_dir_loss(model, net_output, sample, reduce) if model.training else zero

        # Total loss
        loss = kl_loss + self.self_ratio * gauss_loss

        # Sample size for gradient normalisation
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "kl_loss": kl_loss.data,
            "dir_loss": gauss_loss.data,
            "nll_loss": nll_loss.data,
            "ls_loss": ls_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        gauss_loss_sum = sum(log.get("gauss_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ls_loss_sum = sum(log.get("ls_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "gauss_loss", gauss_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ls_loss", ls_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        # Additional metrics for accuracy amoung others
        cls.additional_metrics(logging_outputs)
