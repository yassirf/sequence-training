import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterionConfig(FairseqDataclass):
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
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def gaussian_nll(gaussian_mean: torch.Tensor, gaussian_scale: torch.Tensor, samples: torch.Tensor, ignore_mask=None, reduce=True):
    # Get sufficient statistics
    sample_mean = samples.mean(dim = 1)
    sample_mean_sq = (samples ** 2).mean(dim = 1)

    # Get additional quantities
    gaussian_mean_sq = gaussian_mean ** 2

    # Compute the negative log-likelihood loss
    loss = 2 * gaussian_scale.log() + math.log(2 * math.pi)
    loss += (sample_mean_sq - 2 * gaussian_mean * sample_mean + gaussian_mean_sq)/gaussian_scale**2
    loss = 0.50 * loss.sum(-1)

    # Mask out padding elements
    if ignore_mask is not None: loss.masked_fill_(ignore_mask, 0.0)

    if reduce: loss = loss.sum()
    return loss


@register_criterion(
    "label_smoothed_cross_entropy_and_self_gaussian_nll", dataclass=LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterionConfig
)
class LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            self_ratio=0.0,
            temperature_scale_est=1.0,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            label_smoothing=label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy
        )
        self.self_ratio = self_ratio
        self.use_ratio = self_ratio > 1e-9
        self.temperature_scale_est = temperature_scale_est

    @classmethod
    def build_criterion(cls, cfg: LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterionConfig, task):
        return LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion(
            task=task,
            sentence_avg=cfg.sentence_avg,
            label_smoothing=cfg.label_smoothing,
            ignore_prefix_size=cfg.ignore_prefix_size,
            report_accuracy=cfg.report_accuracy,
            self_ratio=cfg.self_ratio,
            temperature_scale_est=cfg.temperature_scale_est,
        )

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

        # Get logits and additional information
        z, extra = net_output

        # Compute label smoothed loss and nll
        ls_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # Sample size for gradient normalisation
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output = {
            "loss": ls_loss.data,
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

        if model.decoder.num_passes < 1 or \
                not self.use_ratio or \
                not model.training or \
                not 'teacher_predictions_lp' in extra:
            return ls_loss, sample_size, logging_output

        """
        Perform the self loss procedure.
        First we estimate a target dirichlet distribution and compute the KL loss
        """

        # First get teacher/student predictions
        samples = torch.log_softmax(extra['teacher_predictions_lp']/self.temperature_scale_est, dim=-1)
        gaussian_mean = extra['student_predictions_mean']
        gaussian_scale = extra['student_predictions_scale']

        # Get NLL Loss for diagonal gaussians
        self_loss = gaussian_nll(
            gaussian_mean = gaussian_mean,
            gaussian_scale = gaussian_scale,
            samples = samples,
            ignore_mask = sample["target"].eq(self.padding_idx),
            reduce = reduce,
        )

        # Get weighted sum loss
        loss = ls_loss + self.self_ratio * self_loss

        # Update logs
        logging_output["loss"] = loss.data
        logging_output["self_loss"] = self_loss.data

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ls_loss_sum = sum(log.get("ls_loss", 0) for log in logging_outputs)
        self_loss_sum = sum(log.get("self_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ls_loss", ls_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "self_loss", self_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True