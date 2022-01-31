import math
from dataclasses import dataclass, field

import torch
from torch.distributions.dirichlet import Dirichlet
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.dataclass import FairseqDataclass
from .distillation import KLDivergenceCriterionConfig, KLDivergenceCriterion
from self_distribution_distillation_src.utils.dirichlet import DirichletEstimation
from omegaconf import II


@dataclass
class KLDivergenceAndDirCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    ls_ratio: float = field(
        default=0.0,
        metadata={"help": "Weighting of label smoothed loss"}
    )
    self_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio of default to self loss"}
    )
    temperature_scale_est: float = field(
        default=1.0,
        metadata={"help": "temperature scaling teacher predictions for dirichlet estimation and in distillation"}
    )
    temperature_scale_num: float = field(
        default=1.0,
        metadata={"help": "temperature scaling alphas in kl divergence for numerical stability and for distillation"}
    )
    estimation_iter: int = field(
        default=0,
        metadata={"help": "number of iterative steps used to estimate a target dirichlet"}
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


def dirichlet_kl_divergence(log_alphas, teacher_log_alphas, temperature_scale_num, ignore_mask=None, reduce=True):

    # Get target scaled distributions
    teacher_alphas = torch.exp(teacher_log_alphas / temperature_scale_num)
    teacher_alphas = Dirichlet(teacher_alphas)

    # Get prediction scaled distribution
    alphas = torch.exp(log_alphas / temperature_scale_num)
    alphas = Dirichlet(alphas)

    # Use built in kl divergence (batch, seq)
    loss = torch.distributions.kl.kl_divergence(teacher_alphas, alphas)

    # Mask out padding elements
    if ignore_mask is not None: loss.masked_fill_(ignore_mask, 0.0)

    if reduce: loss = loss.sum()
    return loss


@register_criterion("kl_divergence_distillation_and_dir", dataclass=KLDivergenceAndDirCriterionConfig)
class KLDivergenceAndDirCriterion(KLDivergenceCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            ls_ratio=0.0,
            self_ratio=0.0,
            temperature_scale_est=1.0,
            temperature_scale_num=1.0,
            estimation_iter=1
    ):
        super(KLDivergenceAndDirCriterion, self).__init__(
            task = task,
            sentence_avg = sentence_avg,
            label_smoothing = label_smoothing,
            ignore_prefix_size = ignore_prefix_size,
            report_accuracy = report_accuracy,
            temperature_scale_est = temperature_scale_est,
            temperature_scale_num = temperature_scale_num,
        )

        # For weighting label smoothed loss
        self.ls_ratio = ls_ratio

        # For dirichlet estimation
        self.self_ratio = self_ratio
        self.estimation_iter = estimation_iter

    @classmethod
    def build_criterion(cls, cfg: KLDivergenceAndDirCriterionConfig, task):
        return KLDivergenceAndDirCriterion(
            task = task,
            sentence_avg = cfg.sentence_avg,
            label_smoothing = cfg.label_smoothing,
            ignore_prefix_size = cfg.ignore_prefix_size,
            report_accuracy = cfg.report_accuracy,
            ls_ratio = cfg.ls_ratio,
            self_ratio=cfg.self_ratio,
            temperature_scale_est = cfg.temperature_scale_est,
            temperature_scale_num = cfg.temperature_scale_num,
            estimation_iter = cfg.estimation_iter,
        )

    def compute_dir_loss(self, model, net_output, sample, reduce=True):
        """
        Estimate and compute the kl-divergence between student and teacher dirichlets
        """

        with torch.no_grad():
            # Get teacher predictions
            teacher_log_probs = sample['teacher_ensemble_logits']/self.temperature_scale_est
            teacher_log_probs = torch.log_softmax(teacher_log_probs, dim = -1)

            # Estimate target predictions
            teacher_log_alphas = DirichletEstimation(
                logprobs = teacher_log_probs.transpose(1, 2),
                estimation_iter = self.estimation_iter
            ).estimation()

        # Get student predictions
        log_alphas = net_output[0]

        # Compute loss
        loss = dirichlet_kl_divergence(
            log_alphas = log_alphas,
            teacher_log_alphas = teacher_log_alphas,
            temperature_scale_num = self.temperature_scale_num,
            ignore_mask = self.get_padding_mask(sample),
            reduce = reduce,
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

        # Get label smoothed and nll loss
        ls_loss, nll_loss = self.compute_nll_loss(model, net_output, sample, reduce)

        # Zero element
        zero = torch.zeros_like(ls_loss)

        # Get kl-divergence loss only during training
        kl_loss = self.compute_kl_loss(model, net_output, sample, reduce) if model.training else zero

        # Get dirichlet loss only during training
        dir_loss = self.compute_dir_loss(model, net_output, sample, reduce) if model.training else zero

        # Total loss
        loss = ls_loss * self.ls_ratio + (kl_loss + self.self_ratio * dir_loss) * (1 - self.ls_ratio)

        # Sample size for gradient normalisation
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "kl_loss": kl_loss.data,
            "dir_loss": dir_loss.data,
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
        dir_loss_sum = sum(log.get("dir_loss", 0) for log in logging_outputs)
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
            "dir_loss", dir_loss_sum / sample_size / math.log(2), sample_size, round=3
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
