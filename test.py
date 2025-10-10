import time
import os
import torch  # <-- precisava deste import
# from torch import nn
import torch.optim  # ok manter
from torch.utils import data
import argparse
import json

from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    """
    Testing.
    """
    # CPU only
    device = torch.device('cpu')

    os.makedirs(args.savepath, exist_ok=True)
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    # Load checkpoint (CPU)
    snapshot_full_path = os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(snapshot_full_path, map_location=device)

    # Build models
    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(
        n_layers=args.n_layers,
        feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
        heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout
    )
    decoder = DecoderTransformer(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_vocab),
        max_lengths=args.max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])

    # Eval + mover para CPU
    encoder.eval().to(device)
    encoder_trans.eval().to(device)
    decoder.eval().to(device)

    # Dataloaders
    pin = False
    if args.data_name == 'LEVIR_CC':
        nochange_list = [
            "the scene is the same as before ", "there is no difference ",
            "the two scenes seem identical ", "no change has occurred ",
            "almost nothing has changed "
        ]
        test_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=pin
        )
    elif args.data_name == 'Dubai_CC':
        nochange_list = [
            "Nothing has changed ", "There is no difference ", "All remained the same ",
            "Everything remains the same ", "Nothing has changed in this area ", "Nothing changed in this area ",
            "No changes in this area ", "No change was made ", "There is no change to mention ", "No changes to mention ",
            "No changed to mention ", "No difference in this area ", "No change to mention ",
            "No change was made ", "No change occurred in this area ", "The area appears the same "
        ]
        test_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=pin
        )
    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    # Test loop
    test_start_time = time.time()
    references, hypotheses = [], []
    change_references, change_hypotheses = [], []
    nochange_references, nochange_hypotheses = [], []
    change_acc = 0
    nochange_acc = 0

    with torch.no_grad():
        for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate(test_loader):
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            if args.data_name == 'Dubai_CC':
                imgA = l_resize1(imgA)
                imgB = l_resize2(imgB)
            token_all = token_all.squeeze(0).to(device, dtype=torch.long)

            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)
            seq = decoder.sample(feat1, feat2)

            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)

            pred_caption = " ".join(list(word_vocab.keys())[i] for i in pred_seq) + (" " if len(pred_seq) else "")
            ref_caption = " ".join(list(word_vocab.keys())[i] for i in img_tokens[0]) + " "

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc += 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc += 1

        test_time = time.time() - test_start_time

        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            print('BLEU-1: {0:.4f}\tBLEU-2: {1:.4f}\tBLEU-3: {2:.4f}\tBLEU-4: {3:.4f}\tMeteor: {4:.4f}\tRouge: {5:.4f}\tCider: {6:.4f}\t'
                  .format(nochange_metric['Bleu_1'], nochange_metric['Bleu_2'], nochange_metric['Bleu_3'], nochange_metric['Bleu_4'],
                          nochange_metric['METEOR'], nochange_metric['ROUGE_L'], nochange_metric['CIDEr']))
            print("nochange_acc:", nochange_acc / len(nochange_references))

        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            print('BLEU-1: {0:.4f}\tBLEU-2: {1:.4f}\tBLEU-3: {2:.4f}\tBLEU-4: {3:.4f}\tMeteor: {4:.4f}\tRouge: {5:.4f}\tCider: {6:.4f}\t'
                  .format(change_metric['Bleu_1'], change_metric['Bleu_2'], change_metric['Bleu_3'], change_metric['Bleu_4'],
                          change_metric['METEOR'], change_metric['ROUGE_L'], change_metric['CIDEr']))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        print('Testing:\nTime: {0:.3f}\tBLEU-1: {1:.4f}\tBLEU-2: {2:.4f}\tBLEU-3: {3:.4f}\tBLEU-4: {4:.4f}\tMeteor: {5:.4f}\tRouge: {6:.4f}\tCider: {7:.4f}\t'
              .format(test_time, score_dict['Bleu_1'], score_dict['Bleu_2'], score_dict['Bleu_3'], score_dict['Bleu_4'],
                      score_dict['METEOR'], score_dict['ROUGE_L'], score_dict['CIDEr']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Rotas locais (ajuste se preciso)
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='path of the data lists')
    parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')

    # Nome EXATO do arquivo salvo por train.py (ajuste aqui!)
    parser.add_argument('--checkpoint', default='auto', help='checkpoint filename inside savepath or "auto" to infer from data_name')

    # Dubai (exemplo)
    # parser.add_argument('--data_folder', default='./data/Dubai_CC/imgs_tiles/RGB/', help='folder with data files')
    # parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    # parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    # parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    # parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    # parser.add_argument('--allow_unk', type=int, default=0, help='path of the data lists')
    # parser.add_argument('--data_name', default="Dubai_CC", help='base name shared by data files.')

    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features: resnet101, vgg16')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; 0 é seguro em Windows/CPU.')
    parser.add_argument('--encoder_dim', type=int, default=2048, help='dimension of extracted features')
    parser.add_argument('--feat_size', type=int, default=16, help='output spatial size of encoder features')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--test_batchsize', type=int, default=1, help='batch_size for testing')
    parser.add_argument('--savepath', default="./models_checkpoint/")

    args = parser.parse_args()

    if args.checkpoint.lower() == 'auto':
        if args.data_name == 'LEVIR_CC':
            args.checkpoint = 'LEVIR_CC_batchsize_32_resnet101.pth'
        elif args.data_name == 'Dubai_CC':
            args.checkpoint = 'Dubai_CC_batchsize_32_resnet101.pth'
        else:
            raise ValueError(f'Cannot infer checkpoint for data_name "{args.data_name}". Please provide --checkpoint manually.')
    main(args)
