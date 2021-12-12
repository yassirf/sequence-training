
from fairseq.models import register_model, register_model_architecture
from examples.speech_recognition.models.vggtransformer import (
    VGGTransformerModel as UserVGGTransformerModel,
    vggtransformer_0,
    vggtransformer_1,
    vggtransformer_2,
    vggtransformer_base,
)

@register_model('speech_vggtransformer')
class VGGTransformerModel(UserVGGTransformerModel):
    @classmethod
    def add_args(cls, parser):
        UserVGGTransformerModel.add_args(parser)


@register_model_architecture('speech_vggtransformer', 'speech_vggtransformer_0')
def speech_vggtransformer_0(args):
    vggtransformer_0(args)


@register_model_architecture('speech_vggtransformer', 'speech_vggtransformer_1')
def speech_vggtransformer_1(args):
    vggtransformer_1(args)


@register_model_architecture('speech_vggtransformer', 'speech_vggtransformer_2')
def speech_vggtransformer_2(args):
    vggtransformer_2(args)


@register_model_architecture('speech_vggtransformer', 'speech_vggtransformer_base')
def speech_vggtransformer_base(args):
    vggtransformer_base(args)