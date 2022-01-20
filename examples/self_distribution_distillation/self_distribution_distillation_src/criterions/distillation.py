import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class KLDivergenceCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    temperature_scale_est: float = field(
        default=1.0,
        metadata={"help": "temperature scaling teacher predictions"}
    )
    temperature_scale_num: float = field(
        default=1.0,
        metadata={"help": "symmetric temperature scaling for numerical stability"}
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


def kl_divergence(log_probs, teacher_log_probs, ignore_mask = None, reduce = True):
    """
    The inputs will have shape
    log_probs:           (batch, len, vocab)
    teacher_log_probs:   (batch, len, models, vocab)
    """
    # Matching the shape of the target
    log_probs = log_probs.unsqueeze(dim = 2)

    # Compute kl-divergence between categoricals
    loss = torch.exp(teacher_log_probs) * (teacher_log_probs - log_probs)

    # Sum over vocabulary and average over all teacher members
    loss = loss.sum(dim = -1).mean(dim = 2)

    # Mask padding elements
    if ignore_mask is not None: loss.masked_fill_(ignore_mask, 0.0)

    if reduce: loss = loss.sum()
    return loss


@register_criterion("kl_divergence_distillation", dataclass=KLDivergenceCriterionConfig)
class KLDivergenceCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            temperature_scale_est = 1.0,
            temperature_scale_num = 1.0,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            label_smoothing=label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy
        )

        # When obtaining teacher probabilities
        self.temperature_scale_est = temperature_scale_est

        # Symmetric temperature in kl-loss
        self.temperature_scale_num = temperature_scale_num

    @classmethod
    def build_criterion(cls, cfg: KLDivergenceCriterionConfig, task):
        return KLDivergenceCriterion(
            task = task,
            sentence_avg = cfg.sentence_avg,
            label_smoothing = cfg.label_smoothing,
            ignore_prefix_size = cfg.ignore_prefix_size,
            report_accuracy = cfg.report_accuracy,
            temperature_scale_est = cfg.temperature_scale_est,
            temperature_scale_num = cfg.temperature_scale_num,
        )

    def get_padding_mask(self, sample):
        return sample["target"].eq(self.padding_idx)

    @torch.no_grad()
    def compute_nll_loss(self, model, net_output, sample, reduce=True):
        """
        Compute the smooth and negative log-likelihood during validation for tracking purposes
        """
        return super(KLDivergenceCriterion, self).compute_loss(
            model, net_output, sample, reduce = reduce
        )

    def compute_kl_loss(self, model, net_output, sample, reduce=True):
        """
        Compute the expected kl-divergence between student and teacher parameters
        """

        # Get student predictions
        log_probs = net_output[0]/self.temperature_scale_num
        log_probs = torch.log_softmax(log_probs, dim = -1)

        with torch.no_grad():
            # Get teacher predictions
            teacher_log_probs = sample['teacher_ensemble_logits']/(self.temperature_scale_num * self.temperature_scale_est)
            teacher_log_probs = torch.log_softmax(teacher_log_probs, dim = -1)

        # Get the kl-divergence loss
        loss = kl_divergence(
            log_probs = log_probs,
            teacher_log_probs = teacher_log_probs,
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

        # Get kl-divergence loss only during training
        loss = self.compute_kl_loss(model, net_output, sample, reduce) if model.training else 0.0

        # Sample size for gradient normalisation
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        logging_output = {
            "loss": loss.data,
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
    def additional_metrics(cls, logging_outputs) -> None:
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

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """
        Aggregate logging outputs from data parallel training.
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ls_loss_sum = sum(log.get("ls_loss", 0) for log in logging_outputs)
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
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        # Additional metrics for accuracy amoung others
        cls.additional_metrics(logging_outputs)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True