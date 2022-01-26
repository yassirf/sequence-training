import math
import torch
from typing import Optional
from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairseq import utils
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big
)

from .utils import mimo_batchify
from .embedding import MimoEmbedding
from .encoder import NaiveMimoTransformerEncoder
from .decoder import NaiveMimoTransformerDecoder


@register_model('naive_mimo_transformer')
class NaiveMimoTransformerModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--mimo-num-heads', type=int, default=2)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        cfg = TransformerConfig.from_namespace(args)

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = MimoEmbedding(args.mimo_num_heads, num_embeddings, embed_dim, padding_idx = None)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            for i in range(args.mimo_num_heads):
                utils.load_embedding(embed_dict, dictionary, emb.embs[i])
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return NaiveMimoTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        cfg = TransformerConfig.from_namespace(args)
        return NaiveMimoTransformerDecoder(args, tgt_dict, embed_tokens, no_encoder_attn = cfg.no_cross_attention)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        # At inference time we repeat the input several times
        if not self.training:
            src_tokens = torch.cat([src_tokens] * self.args.mimo_num_heads)
            prev_output_tokens = torch.cat([prev_output_tokens] * self.args.mimo_num_heads)

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        # Get the decoder outputs
        decoder_x, decoder_extra = decoder_out

        # Get the predictions and store in a key: teacher_predictions_lp
        if not self.training:
            # The batch size is (models, batch, len, vocab)
            decoder_x = mimo_batchify(decoder_x, self.args.mimo_num_heads)

            # Reorder output into (batch, len, models, vocab)
            decoder_extra['teacher_predictions_lp'] = decoder_x.permute(1, 2, 0, 3)

            # Get the ensemble prediction of heads
            decoder_x = torch.log_softmax(decoder_x, dim = -1)
            decoder_x = torch.logsumexp(decoder_x, dim = 0) - math.log(decoder_x.size(0))

        return decoder_x, decoder_extra


@register_model_architecture('naive_mimo_transformer', 'naive_mimo_transformer')
def naive_mimo_transformer(args):
    base_architecture(args)


@register_model_architecture('naive_mimo_transformer', 'naive_mimo_transformer_wmt_en_de_big')
def naive_mimo_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
