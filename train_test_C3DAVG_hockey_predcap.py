# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import json
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.S2VTModel_hockey import S2VTModel
from opts import *
from utils import utils_1
import numpy as np
import nltk
import matplotlib.pyplot as plt

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True


def decode_sentence(sentence):
    info = json.load(open(os.path.join(anno_n_splits_dir, 'vocab.json'), 'rb'))
    ix_to_word = info['ix_to_word']

    # batches, sentence_len = sentence.shape
    # decode_sentence = np.empty((batches, sentence_len), dtype = np.dtype(str))
    # decode_sentences = []

    # for b in range(batches):
    decode_sentence = []
    for j, w in enumerate(sentence):
        if ix_to_word[str(int(w))] == '<eos>':
            # decode_sentence.append(ix_to_word[str(int(w.cpu()))])
            decode_sentence.append(ix_to_word[str(int(w))])
            break
        # decode_sentence.append(ix_to_word[str(int(w.cpu()))])
        decode_sentence.append(ix_to_word[str(int(w))])
        # decode_sentences.append(decode_sentence)

    return decode_sentence


def save_model(model, model_name, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)

    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    criterion_hockey_classifier = criterions['criterion_hockey_classifier']
    criterion_caption = criterions['criterion_caption']

    model_CNN.train()
    model_my_fc6.train()
    model_caption.train()

    iteration = 0
    for data in train_dataloader:
        # Hockey classification
        true_switch = data['label_switch'].cuda()
        true_advance = data['label_advance'].cuda()
        true_faceoff = data['label_faceoff'].cuda()
        true_play_make = data['label_play_make'].cuda()
        true_play_receive = data['label_play_receive'].cuda()
        # true_whistle = data['label_whistle'].cuda()
        true_shot = data['label_shot'].cuda()
        # true_hit = data['label_hit'].cuda() 
        # true_shot_block = data['label_shot_block'].cuda() 
        # true_penalty = data['label_penalty'].cuda() 
        # true_ricochet = data['label_ricochet'].cuda()

        # Captions
        true_captions = data['label_captions'].cuda()
        true_captions_mask = data['label_captions_mask'].cuda()

        video = data['video'].transpose_(1, 2).cuda()

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([]).cuda()

        for i in np.arange(0, frames - 17, 16):
            clip = video[:, :, i:i + 16, :, :]
            clip_feats_temp = model_CNN(clip)
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
        clip_feats_avg = clip_feats.mean(1)

        sample_feats_fc6 = model_my_fc6(clip_feats_avg)

        # (seq_probs, _, pred_switch, pred_advance, pred_faceoff, pred_play_make, 
        #     pred_play_receive, pred_whistle, pred_shot, pred_hit, 
        #     pred_shot_block, pred_penalty, pred_ricochet) = model_caption(clip_feats, sample_feats_fc6, true_captions, 'train')
        (seq_probs, _, pred_switch, pred_advance, pred_faceoff, pred_play_make, 
            pred_play_receive, pred_shot) = model_caption(clip_feats, sample_feats_fc6, true_captions, 'train')

        loss = 0

        total_occurences = sum(list(class_occurences.values()))
        class_weights = {}
        for k, v in class_occurences.items():
            class_weights[k] = total_occurences / v
        weights_sum = sum(list(class_weights.values()))
        for k, v in class_weights.items():
            class_weights[k] = v / weights_sum

        loss_switch = class_weights['SwitchEvent'] * criterion_hockey_classifier(pred_switch, true_switch)
        loss_advance = class_weights['AdvanceEvent'] * criterion_hockey_classifier(pred_advance, true_advance)
        loss_faceoff = class_weights['FaceoffEvent'] * criterion_hockey_classifier(pred_faceoff, true_faceoff)
        loss_play_make = class_weights['PlayMakeEvent'] * criterion_hockey_classifier(pred_play_make, true_play_make)
        loss_play_receive = class_weights['PlayReceiveEvent'] * criterion_hockey_classifier(pred_play_receive, true_play_receive)
        # loss_whistle = criterion_hockey_classifier(pred_whistle, true_whistle)
        loss_shot = class_weights['ShotEvent'] * criterion_hockey_classifier(pred_shot, true_shot)
        # loss_hit = criterion_hockey_classifier(pred_hit, true_hit)
        # loss_shot_block = criterion_hockey_classifier(pred_shot_block, true_shot_block)
        # loss_penalty = criterion_hockey_classifier(pred_penalty, true_penalty)
        # loss_ricochet = criterion_hockey_classifier(pred_ricochet, true_ricochet)
        loss_cls = loss_switch + loss_advance + loss_faceoff + loss_play_make
        loss_cls += loss_play_receive + loss_shot
        # loss_cls += loss_shot_block + loss_penalty + loss_ricochet
        loss += loss_cls

        loss_caption = criterion_caption(seq_probs, true_captions[:, 1:], true_captions_mask[:, 1:])
        loss += loss_caption * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # softmax_layer = nn.Softmax(dim = 1)
        # temp_caption = softmax_layer(seq_probs).data.cpu().numpy()

        # pred_captions = []
        # true_captions = []
        # label_captions = []

        # true_captions.extend(data['label_captions'].cuda())

        # for i in range(len(temp_caption)):
        # #     print(np.argmax(temp_switch[i]) == true_switch[i].cpu().numpy())
        # #     print(true_switch[i].cpu().numpy())
        #     pred_captions.append(decode_sentence(np.argmax(temp_caption[i], axis = 1)))
        #     label_captions.append(decode_sentence(true_captions[i]))

        # print(pred_captions)
        # print(label_captions)

        # print(nltk.translate.bleu_score.corpus_bleu(label_captions, pred_captions))

        # exit()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, end="")
            print(' Cap Loss: ', loss_caption, end="")
            print(' ')
        iteration += 1

    return loss


def test_phase(test_dataloader):
    print('In testphase...')

    with torch.no_grad():
        pred_switch = []; pred_advance = []; pred_faceoff = []; pred_play_make = [] 
        pred_play_receive = []; pred_shot = []
        # pred_whistle = []; pred_hit = []
        # pred_shot_block = []; pred_penalty = []; pred_ricochet = []
        true_switch = []; true_advance = []; true_faceoff = []; true_play_make = []
        true_play_receive = []; true_shot = []
        # true_whistle = []; true_hit = []
        # true_shot_block = []; true_penalty = []; true_ricochet = []
        pred_captions = []; true_captions = []; label_captions = []

        model_CNN.eval()
        model_my_fc6.eval()
        model_caption.eval()

        for data in test_dataloader:
            # Hockey classification
            true_switch.extend(data['label_switch'].numpy())
            true_advance.extend(data['label_advance'].numpy())
            true_faceoff.extend(data['label_faceoff'].numpy())
            true_play_make.extend(data['label_play_make'].numpy())
            true_play_receive.extend(data['label_play_receive'].numpy())
            # true_whistle.extend(data['label_whistle'].numpy())
            true_shot.extend(data['label_shot'].numpy())
            # true_hit.extend(data['label_hit'].numpy())
            # true_shot_block.extend(data['label_shot_block'].numpy())
            # true_penalty.extend(data['label_penalty'].numpy())
            # true_ricochet.extend(data['label_ricochet'].numpy())

            # Captions
            true_captions.extend(data['label_captions'].cuda())
            true_caption = data['label_captions'].cuda()

            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
                clip_feats_temp = model_CNN(clip)
                clip_feats_temp.unsqueeze_(0)
                clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
            clip_feats_avg = clip_feats.mean(1)

            sample_feats_fc6 = model_my_fc6(clip_feats_avg)

            # (seq_probs, seq_preds, temp_switch, temp_advance, temp_faceoff, temp_play_make, 
            #     temp_play_receive, temp_whistle, temp_shot, temp_hit, 
            #     temp_shot_block, temp_penalty, temp_ricochet) = model_caption(clip_feats, sample_feats_fc6, true_caption, mode = 'train')
            (seq_probs, seq_preds, temp_switch, temp_advance, temp_faceoff, temp_play_make, 
                temp_play_receive, temp_shot) = model_caption(clip_feats, sample_feats_fc6, true_caption, mode = 'train')

            softmax_layer = nn.Softmax(dim = 1)
            temp_switch = softmax_layer(temp_switch).data.cpu().numpy()
            temp_advance = softmax_layer(temp_advance).data.cpu().numpy()
            temp_faceoff = softmax_layer(temp_faceoff).data.cpu().numpy()
            temp_play_make = softmax_layer(temp_play_make).data.cpu().numpy()
            temp_play_receive = softmax_layer(temp_play_receive).data.cpu().numpy()
            # temp_whistle = softmax_layer(temp_whistle).data.cpu().numpy()
            temp_shot = softmax_layer(temp_shot).data.cpu().numpy()
            # temp_hit = softmax_layer(temp_hit).data.cpu().numpy()
            # temp_shot_block = softmax_layer(temp_shot_block).data.cpu().numpy()
            # temp_penalty = softmax_layer(temp_penalty).data.cpu().numpy()
            # temp_ricochet = softmax_layer(temp_ricochet).data.cpu().numpy()

            temp_caption = softmax_layer(seq_probs).data.cpu().numpy()

            for i in range(len(temp_switch)):
                pred_switch.append(np.argmax(temp_switch[i]))
                pred_advance.append(np.argmax(temp_advance[i]))
                pred_faceoff.append(np.argmax(temp_faceoff[i]))
                pred_play_make.append(np.argmax(temp_play_make[i]))
                pred_play_receive.append(np.argmax(temp_play_receive[i]))
                # pred_whistle.append(np.argmax(temp_whistle[i]))
                pred_shot.append(np.argmax(temp_shot[i]))
                # pred_hit.append(np.argmax(temp_hit[i]))
                # pred_shot_block.append(np.argmax(temp_shot_block[i]))
                # pred_penalty.append(np.argmax(temp_penalty[i]))
                # pred_ricochet.append(np.argmax(temp_ricochet[i]))
                # pred_ricochet.extend(np.argwhere(true_ricochet[i] == max(temp_ricochet[i])))

                pred_captions.append(decode_sentence(np.argmax(temp_caption[i], axis = 1)))
                label_captions.append(decode_sentence(true_captions[i]))
            

        # correct_class = {'switch_correct': 0, 'advance_correct': 0, 'faceoff_correct': 0, 'play_make_correct': 0,
        #     'play_receive_correct': 0, 'whistle_correct': 0, 'shot_correct': 0, 'hit_correct': 0, 'shot_block_correct': 0, 
        #     'penalty_correct': 0, 'ricochet_correct': 0}
            correct_class = {'switch_correct': 0, 'advance_correct': 0, 'faceoff_correct': 0, 'play_make_correct': 0,
            'play_receive_correct': 0, 'shot_correct': 0}
        for i in range(len(pred_switch)):
            if pred_switch[i] == true_switch[i]:
                correct_class['switch_correct'] += 1
            if pred_advance[i] == true_advance[i]:
                correct_class['advance_correct'] += 1
            if pred_faceoff[i] == true_faceoff[i]:
                correct_class['faceoff_correct'] += 1
            if pred_play_make[i] == true_play_make[i]:
                correct_class['play_make_correct'] += 1
            if pred_play_receive[i] == true_play_receive[i]:
                correct_class['play_receive_correct'] += 1
            # if pred_whistle[i] == true_whistle[i]:
            #     correct_class['whistle_correct'] += 1
            if pred_shot[i] == true_shot[i]:
                correct_class['shot_correct'] += 1
            # if pred_hit[i] == true_hit[i]:
            #     correct_class['hit_correct'] += 1
            # if pred_shot_block[i] == true_shot_block[i]:
            #     correct_class['shot_block_correct'] += 1
            # if pred_penalty[i] == true_penalty[i]:
            #     correct_class['penalty_correct'] += 1
            # if pred_ricochet[i] == true_ricochet[i]:
            #     correct_class['ricochet_correct'] += 1

        for k, v in correct_class.items():
            print('{}: {}'.format(k, v / len(pred_switch) * 100))

        print('BLEU score:', nltk.translate.bleu_score.corpus_bleu(label_captions, pred_captions))


def main():
    parameters_2_optimize = (list(model_CNN.parameters()) + list(model_my_fc6.parameters())) + list(model_caption.parameters())
    parameters_2_optimize_named = (list(model_CNN.named_parameters()) + list(model_my_fc6.named_parameters())) + list(model_caption.named_parameters())

    optimizer = optim.Adam(parameters_2_optimize, lr = 0.0001)
    print('Parameters that will be learnt:', len(parameters_2_optimize_named))

    criterions = {}
    criterion_hockey_classifier = nn.CrossEntropyLoss()
    criterions['criterion_hockey_classifier'] = criterion_hockey_classifier
    criterion_caption = utils_1.LanguageModelCriterion()
    criterions['criterion_caption'] = criterion_caption

    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    print('Length of train loader: ', len(train_dataloader))
    print('Length of test loader: ', len(test_dataloader))
    print('Training set size: ', len(train_dataloader)*train_batch_size,
          ';    Test set size: ', len(test_dataloader)*test_batch_size)

    losses = []

    # actual training, testing loops
    for epoch in range(100):
        print('---------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        loss = train_phase(train_dataloader, optimizer, criterions, epoch)
        losses.append(loss.cpu().item())
        test_phase(test_dataloader)

        if (epoch + 1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_my_fc6, 'model_my_fc6', epoch, saving_dir)
            save_model(model_caption, 'model_caption', epoch, saving_dir)

    plt.plot(range(len(losses)), losses)
    plt.savefig('losses5.png')


if __name__ == '__main__':
    # loading the altered C3D backbone (ie C3D upto before fc-6)
    model_CNN_pretrained_dict = torch.load('c3d.pickle')
    model_CNN = C3D_altered()
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.cuda()

    model_caption = S2VTModel(vocab_size, max_cap_len, caption_lstm_dim_hidden,
                              caption_lstm_dim_word, caption_lstm_dim_vid,
                              rnn_cell=caption_lstm_cell_type, n_layers=caption_lstm_num_layers,
                              rnn_dropout_p=caption_lstm_dropout,
                              eos_id=1719, sos_id=2060)
    model_caption = model_caption.cuda()
    print('Using Captioning Loss')

    main()