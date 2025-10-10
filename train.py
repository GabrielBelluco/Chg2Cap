import time
import os
import numpy as np
import torch
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
# import torchvision.transforms as transforms
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils import *

def main(args):
    """
    Training and validation.
    """
    # --- CPU only: remover variáveis de ambiente de CUDA ---
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Dispositivo (CPU)
    device = torch.device('cpu')

    if os.path.exists(args.savepath) == False:
        os.makedirs(args.savepath)
    best_bleu4 = 0.4  # BLEU-4 score right now
    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    # Initialize / load checkpoint
    if args.checkpoint is None:
        encoder = Encoder(args.network)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None
        encoder_trans = AttentiveEncoder(n_layers=args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                         heads=args.n_heads, hidden_dim=args.hidden_dim, attention_dim=args.attention_dim, dropout=args.dropout)
        encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                                   lr=args.encoder_lr)
        decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                     n_layers=args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
    else:
        # --- garantir que o checkpoint é carregado na CPU ---
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    # --- mover modelos para CPU ---
    encoder = encoder.to(device)
    encoder_trans = encoder_trans.to(device)
    decoder = decoder.to(device)

    # Loss function (na CPU)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=False)
    elif args.data_name == 'Dubai_CC':
        train_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=False)

    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.5) if args.fine_tune_encoder else None
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_trans_optimizer, step_size=5, gamma=0.5)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    l_resizeA = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resizeB = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))

    # Epochs
    for epoch in range(start_epoch, args.num_epochs):
        # Batches
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_loader):
            start_time = time.time()
            decoder.train()
            encoder.train()
            encoder_trans.train()
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            # --- mover tensores para CPU (device genérico) ---
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            if args.data_name == 'Dubai_CC':
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            token = token.squeeze(1).to(device, dtype=torch.long)
            token_len = token_len.to(device)

            # Forward
            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(feat1, feat2, token, token_len)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            targets = targets.long()

            loss = criterion(scores, targets)
            loss.backward()

            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)

            # Update weights
            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Metrics
            hist[index_i, 0] = time.time() - start_time
            hist[index_i, 1] = loss.item()
            hist[index_i, 2] = accuracy(scores, targets, 5)
            index_i += 1

            # Print status
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time: {3:.3f}\t'
                      'Loss: {4:.4f}\t'
                      'Top-5 Accuracy: {5:.3f}'.format(
                    epoch, index_i, args.num_epochs * len(train_loader),
                    np.mean(hist[index_i - args.print_freq:index_i - 1, 0]) * args.print_freq,
                    np.mean(hist[index_i - args.print_freq:index_i - 1, 1]),
                    np.mean(hist[index_i - args.print_freq:index_i - 1, 2])))

        # Validation
        decoder.eval()
        encoder_trans.eval()
        if encoder is not None:
            encoder.eval()

        val_start_time = time.time()
        references = list()
        hypotheses = list()

        with torch.no_grad():
            for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate(val_loader):
                imgA = imgA.to(device)
                imgB = imgB.to(device)
                if args.data_name == 'Dubai_CC':
                    imgA = l_resizeA(imgA)
                    imgB = l_resizeB(imgB)
                token_all = token_all.squeeze(0).to(device, dtype=torch.long)

                if encoder is not None:
                    feat1, feat2 = encoder(imgA, imgB)
                feat1, feat2 = encoder_trans(feat1, feat2)
                seq = decoder.sample(feat1, feat2, k=1)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                                      img_token))
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "

            val_time = time.time() - val_start_time
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
                  'BLEU-4: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Rouge: {6:.4f}\t' 'Cider: {7:.4f}\t'
                  .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))

        # LR schedulers
        decoder_lr_scheduler.step()
        encoder_trans_lr_scheduler.step()
        if encoder_lr_scheduler is not None:
            encoder_lr_scheduler.step()

        # Save best
        if Bleu_4 > best_bleu4:
            best_bleu4 = max(Bleu_4, best_bleu4)
            print('Save Model')
            state = {
                'encoder_dict': encoder.state_dict(),
                'encoder_trans_dict': encoder_trans.state_dict(),
                'decoder_dict': decoder.state_dict()
            }
            model_name = str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'Bleu_4_'+str(round(10000*best_bleu4))+'.pth'
            torch.save(state, os.path.join(args.savepath, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters (ROTAS LOCAIS)
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')

    # Exemplos para Dubai (mantenho comentado)
    # parser.add_argument('--data_folder', default='./data/Dubai_CC/imgs_tiles/RGB/', help='folder with data files')
    # parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    # parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    # parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    # parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    # parser.add_argument('--allow_unk', type=int, default=0, help='if unknown token is allowed')
    # parser.add_argument('--data_name', default="Dubai_CC", help='base name shared by data files.')

    # GPU id não é usado, mas deixo para compatibilidade de args
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')

    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=32, help='batch_size for training')
    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim', default=2048, help='the dimension of extracted features using different network')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; 0 é seguro em ambientes Windows/CPU.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/", help='where to save checkpoints')

    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)

    args = parser.parse_args()
    main(args)
