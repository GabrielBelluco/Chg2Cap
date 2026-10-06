"""Teacher-forced loss; reference prefixes are not free-generation predictions.

The frozen visual backbone runs without gradients; attentive encoder and decoder
receive gradients. Padding is removed without dropping reference words or EOS.
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence

def trim_padding(tokens, lengths):
    """Remove batch padding only: every reference token, including END, remains."""
    maximum = int(lengths.max())
    if maximum > tokens.shape[1] or int(lengths.min()) < 2:
        raise ValueError('Invalid caption lengths')
    return tokens[:, :maximum]

def prepare_inputs(batch):
    before, after, _, _, tokens, lengths, names = batch
    return before.cuda(), after.cuda(), trim_padding(tokens, lengths).cuda(), lengths.cuda(), names

def batch_loss(encoder, attentive, decoder, batch):
    before, after, tokens, lengths, _ = prepare_inputs(batch)
    with torch.no_grad():
        a, b = encoder(before, after)
    a, b = attentive(a, b)
    scores, captions, decoding_lengths, _ = decoder(a, b, tokens, lengths)
    scores = pack_padded_sequence(scores, decoding_lengths, batch_first=True).data
    targets = pack_padded_sequence(captions[:, 1:], decoding_lengths, batch_first=True).data
    loss = torch.nn.functional.cross_entropy(scores, targets)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss')
    return loss, sum(decoding_lengths)

def validation_loss(encoder, attentive, decoder, loader):
    attentive.eval()
    decoder.eval()
    total, count = 0., 0
    with torch.no_grad():
        for batch in loader:
            loss, n = batch_loss(encoder, attentive, decoder, batch)
            total += loss.item() * n
            count += n
    return total / count
