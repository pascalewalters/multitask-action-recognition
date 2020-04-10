import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.hockey_classifier import hockey_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts import *
from utils import utils_1
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True


def save_model(model, model_name, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)

    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    if with_score_regression:
        criterion_final_score = criterions['criterion_final_score']; 
        penalty_final_score = criterions['penalty_final_score']
    if with_dive_classification:
        criterion_dive_classifier = criterions['criterion_dive_classifier']
    if with_hockey_classification:
        criterion_hockey_classifier = criterions['criterion_hockey_classifier']
    if with_caption:
        # criterion_hockey_classifier = criterions['criterion_hockey_classifier']
        criterion_caption = criterions['criterion_caption']

    model_CNN.train()
    model_my_fc6.train()
    if with_score_regression:
        model_score_regressor.train()
    if with_dive_classification:
        model_dive_classifier.train()
    if with_hockey_classification:
        model_hockey_classifier.train()
    if with_caption:
        model_caption.train()

    iteration = 0
    for data in train_dataloader:
        if with_score_regression:
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
        if with_dive_classification:
            true_postion = data['label_position'].cuda()
            true_armstand = data['label_armstand'].cuda()
            true_rot_type = data['label_rot_type'].cuda()
            true_ss_no = data['label_ss_no'].cuda()
            true_tw_no = data['label_tw_no'].cuda()
        if with_hockey_classification:
            true_switch = data['label_switch'].cuda()
            true_advance = data['label_advance'].cuda()
            true_faceoff = data['label_faceoff'].cuda()
            true_play_make = data['label_play_make'].cuda()
            true_play_receive = data['label_play_receive'].cuda()
            true_whistle = data['label_whistle'].cuda()
            true_shot = data['label_shot'].cuda()
            true_hit = data['label_hit'].cuda() 
            true_shot_block = data['label_shot_block'].cuda() 
            true_penalty = data['label_penalty'].cuda() 
            true_ricochet = data['label_ricochet'].cuda() 
        if with_caption:
            true_captions = data['label_captions'].cuda()
            true_captions_mask = data['label_captions_mask'].cuda()
        video = data['video'].transpose_(1, 2).cuda()

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([]).cuda()

        for i in np.arange(0, frames - 17, 16):
        # for i in np.arange(frames):
            clip = video[:, :, i:i + 16, :, :]
            clip_feats_temp = model_CNN(clip)
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
        clip_feats_avg = clip_feats.mean(1)

        sample_feats_fc6 = model_my_fc6(clip_feats_avg)

        if with_score_regression:
            pred_final_score = model_score_regressor(sample_feats_fc6)
        if with_dive_classification:
            (pred_position, pred_armstand, pred_rot_type, pred_ss_no,
             pred_tw_no) = model_dive_classifier(sample_feats_fc6)
        if with_hockey_classification:
            (pred_switch, pred_advance, pred_faceoff, pred_play_make, 
                pred_play_receive, pred_whistle, pred_shot, pred_hit, 
                pred_shot_block, pred_penalty, pred_ricochet) = model_hockey_classifier(sample_feats_fc6)
        if with_caption:
            (seq_probs, _, pred_switch, pred_advance, pred_faceoff, pred_play_make, 
                pred_play_receive, pred_whistle, pred_shot, pred_hit, 
                pred_shot_block, pred_penalty, pred_ricochet) = model_caption(clip_feats, sample_feats_fc6, true_captions, 'train')

        loss = 0
        if with_score_regression:
            loss_final_score = (criterion_final_score(pred_final_score, true_final_score)
                            + penalty_final_score(pred_final_score, true_final_score))
            loss += loss_final_score
        if with_dive_classification:
            loss_position = criterion_dive_classifier(pred_position, true_postion)
            loss_armstand = criterion_dive_classifier(pred_armstand, true_armstand)
            loss_rot_type = criterion_dive_classifier(pred_rot_type, true_rot_type)
            loss_ss_no = criterion_dive_classifier(pred_ss_no, true_ss_no)
            loss_tw_no = criterion_dive_classifier(pred_tw_no, true_tw_no)
            loss_cls = loss_position + loss_armstand + loss_rot_type + loss_ss_no + loss_tw_no
            loss += loss_cls
        if with_hockey_classification:
            loss_switch = criterion_hockey_classifier(pred_switch, true_switch)
            loss_advance = criterion_hockey_classifier(pred_advance, true_advance)
            loss_faceoff = criterion_hockey_classifier(pred_faceoff, true_faceoff)
            loss_play_make = criterion_hockey_classifier(pred_play_make, true_play_make)
            loss_play_receive = criterion_hockey_classifier(pred_play_receive, true_play_receive)
            loss_whistle = criterion_hockey_classifier(pred_whistle, true_whistle)
            loss_shot = criterion_hockey_classifier(pred_shot, true_shot)
            loss_hit = criterion_hockey_classifier(pred_hit, true_hit)
            loss_shot_block = criterion_hockey_classifier(pred_shot_block, true_shot_block)
            loss_penalty = criterion_hockey_classifier(pred_penalty, true_penalty)
            loss_ricochet = criterion_hockey_classifier(pred_ricochet, true_ricochet)
            loss_cls = loss_switch + loss_advance + loss_faceoff + loss_play_make
            loss_cls += loss_play_receive + loss_whistle + loss_shot + loss_hit
            loss_cls += loss_shot_block + loss_penalty + loss_ricochet
            loss += loss_cls
        if with_caption:
            # loss_switch = criterion_hockey_classifier(pred_switch, true_switch)
            # loss_advance = criterion_hockey_classifier(pred_advance, true_advance)
            # loss_faceoff = criterion_hockey_classifier(pred_faceoff, true_faceoff)
            # loss_play_make = criterion_hockey_classifier(pred_play_make, true_play_make)
            # loss_play_receive = criterion_hockey_classifier(pred_play_receive, true_play_receive)
            # loss_whistle = criterion_hockey_classifier(pred_whistle, true_whistle)
            # loss_shot = criterion_hockey_classifier(pred_shot, true_shot)
            # loss_hit = criterion_hockey_classifier(pred_hit, true_hit)
            # loss_shot_block = criterion_hockey_classifier(pred_shot_block, true_shot_block)
            # loss_penalty = criterion_hockey_classifier(pred_penalty, true_penalty)
            # loss_ricochet = criterion_hockey_classifier(pred_ricochet, true_ricochet)
            # loss_cls = loss_switch + loss_advance + loss_faceoff + loss_play_make
            # loss_cls += loss_play_receive + loss_whistle + loss_shot + loss_hit
            # loss_cls += loss_shot_block + loss_penalty + loss_ricochet
            # loss += loss_cls
            loss_caption = criterion_caption(seq_probs, true_captions[:, 1:], true_captions_mask[:, 1:])
            loss += loss_caption*0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, end="")
            if with_score_regression:
                print(' FS Loss: ', loss_final_score, end="")
            if with_dive_classification:
                print(' Cls Loss: ', loss_cls, end="")
            if with_hockey_classification:
                print(' Hockey Cls Loss: ', loss_cls, end="")
            if with_caption:
                  print(' Cap Loss: ', loss_caption, end="")
            print(' ')
        iteration += 1

    return loss


def test_phase(test_dataloader):
    print('In testphase...')
    with torch.no_grad():
        if with_score_regression:
            pred_scores = []; true_scores = []
        if with_dive_classification:
            pred_position = []; pred_armstand = []; pred_rot_type = []; pred_ss_no = []; pred_tw_no = []
            true_position = []; true_armstand = []; true_rot_type = []; true_ss_no = []; true_tw_no = []
        if with_hockey_classification:
            pred_switch = []; pred_advance = []; pred_faceoff = []; pred_play_make = [] 
            pred_play_receive = []; pred_whistle = []; pred_shot = []; pred_hit = []
            pred_shot_block = []; pred_penalty = []; pred_ricochet = []
            true_switch = []; true_advance = []; true_faceoff = []; true_play_make = []
            true_play_receive = []; true_whistle = []; true_shot = []; true_hit = []
            true_shot_block = []; true_penalty = []; true_ricochet = []
        if with_caption:
            pred_switch = []; pred_advance = []; pred_faceoff = []; pred_play_make = [] 
            pred_play_receive = []; pred_whistle = []; pred_shot = []; pred_hit = []
            pred_shot_block = []; pred_penalty = []; pred_ricochet = []
            true_switch = []; true_advance = []; true_faceoff = []; true_play_make = []
            true_play_receive = []; true_whistle = []; true_shot = []; true_hit = []
            true_shot_block = []; true_penalty = []; true_ricochet = []

        model_CNN.eval()
        model_my_fc6.eval()
        if with_score_regression:
            model_score_regressor.eval()
        if with_dive_classification:
            model_dive_classifier.eval()
        if with_hockey_classification:
            model_hockey_classifier.eval()
        if with_caption:
            model_caption.eval()

        for data in test_dataloader:
            if with_score_regression:
                true_scores.extend(data['label_final_score'].data.numpy())
            if with_dive_classification:
                true_position.extend(data['label_position'].numpy())
                true_armstand.extend(data['label_armstand'].numpy())
                true_rot_type.extend(data['label_rot_type'].numpy())
                true_ss_no.extend(data['label_ss_no'].numpy())
                true_tw_no.extend(data['label_tw_no'].numpy())
            if with_hockey_classification:
            # if with_caption:
                true_switch.extend(data['label_switch'].numpy())
                true_advance.extend(data['label_advance'].numpy())
                true_faceoff.extend(data['label_faceoff'].numpy())
                true_play_make.extend(data['label_play_make'].numpy())
                true_play_receive.extend(data['label_play_receive'].numpy())
                true_whistle.extend(data['label_whistle'].numpy())
                true_shot.extend(data['label_shot'].numpy())
                true_hit.extend(data['label_hit'].numpy())
                true_shot_block.extend(data['label_shot_block'].numpy())
                true_penalty.extend(data['label_penalty'].numpy())
                true_ricochet.extend(data['label_ricochet'].numpy())
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
            if with_score_regression:
                temp_final_score = model_score_regressor(sample_feats_fc6)
                pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
            if with_dive_classification:
                temp_position, temp_armstand, temp_rot_type, temp_ss_no, temp_tw_no = model_dive_classifier(sample_feats_fc6)
                softmax_layer = nn.Softmax(dim=1)
                temp_position = softmax_layer(temp_position).data.cpu().numpy()
                temp_armstand = softmax_layer(temp_armstand).data.cpu().numpy()
                temp_rot_type = softmax_layer(temp_rot_type).data.cpu().numpy()
                temp_ss_no = softmax_layer(temp_ss_no).data.cpu().numpy()
                temp_tw_no = softmax_layer(temp_tw_no).data.cpu().numpy()

                for i in range(len(temp_position)):
                    pred_position.extend(np.argwhere(temp_position[i] == max(temp_position[i]))[0])
                    pred_armstand.extend(np.argwhere(temp_armstand[i] == max(temp_armstand[i]))[0])
                    pred_rot_type.extend(np.argwhere(temp_rot_type[i] == max(temp_rot_type[i]))[0])
                    pred_ss_no.extend(np.argwhere(temp_ss_no[i] == max(temp_ss_no[i]))[0])
                    pred_tw_no.extend(np.argwhere(temp_tw_no[i] == max(temp_tw_no[i]))[0])

            if with_hockey_classification:
            # if with_caption:
                # true_captions = data['label_captions'].cuda()
                # true_captions_mask = data['label_captions_mask'].cuda()
                (temp_switch, temp_advance, temp_faceoff, temp_play_make, 
                    temp_play_receive, temp_whistle, temp_shot, temp_hit, 
                    temp_shot_block, temp_penalty, temp_ricochet) = model_hockey_classifier(sample_feats_fc6)
                # (seq_probs, _, temp_switch, temp_advance, temp_faceoff, temp_play_make, 
                #     temp_play_receive, temp_whistle, temp_shot, temp_hit, 
                #     temp_shot_block, temp_penalty, temp_ricochet) = model_caption(clip_feats, sample_feats_fc6, true_captions, 'train')
                softmax_layer = nn.Softmax(dim = 1)
                temp_switch = softmax_layer(temp_switch).data.cpu().numpy()
                temp_advance = softmax_layer(temp_advance).data.cpu().numpy()
                temp_faceoff = softmax_layer(temp_faceoff).data.cpu().numpy()
                temp_play_make = softmax_layer(temp_play_make).data.cpu().numpy()
                temp_play_receive = softmax_layer(temp_play_receive).data.cpu().numpy()
                temp_whistle = softmax_layer(temp_whistle).data.cpu().numpy()
                temp_shot = softmax_layer(temp_shot).data.cpu().numpy()
                temp_hit = softmax_layer(temp_hit).data.cpu().numpy()
                temp_shot_block = softmax_layer(temp_shot_block).data.cpu().numpy()
                temp_penalty = softmax_layer(temp_penalty).data.cpu().numpy()
                temp_ricochet = softmax_layer(temp_ricochet).data.cpu().numpy()

                for i in range(len(temp_switch)):
                    pred_switch.extend(np.argwhere(temp_switch[i] == max(temp_switch[i]))[0])
                    pred_advance.extend(np.argwhere(temp_advance[i] == max(temp_advance[i]))[0])
                    pred_faceoff.extend(np.argwhere(temp_faceoff[i] == max(temp_faceoff[i]))[0])
                    pred_play_make.extend(np.argwhere(temp_play_make[i] == max(temp_play_make[i]))[0])
                    pred_play_receive.extend(np.argwhere(temp_play_receive[i] == max(temp_play_receive[i]))[0])
                    pred_whistle.extend(np.argwhere(temp_whistle[i] == max(temp_whistle[i]))[0])
                    pred_shot.extend(np.argwhere(temp_shot[i] == max(temp_shot[i]))[0])
                    pred_hit.extend(np.argwhere(temp_hit[i] == max(temp_hit[i]))[0])
                    pred_shot_block.extend(np.argwhere(temp_shot_block[i] == max(temp_shot_block[i]))[0])
                    pred_penalty.extend(np.argwhere(temp_penalty[i] == max(temp_penalty[i]))[0])
                    pred_ricochet.extend(np.argwhere(temp_ricochet[i] == max(temp_ricochet[i]))[0])

        if with_dive_classification:
            position_correct = 0; armstand_correct = 0; rot_type_correct = 0; ss_no_correct = 0; tw_no_correct = 0
            for i in range(len(pred_position)):
                if pred_position[i] == true_position[i]:
                    position_correct += 1
                if pred_armstand[i] == true_armstand[i]:
                    armstand_correct += 1
                if pred_rot_type[i] == true_rot_type[i]:
                    rot_type_correct += 1
                if pred_ss_no[i] == true_ss_no[i]:
                    ss_no_correct += 1
                if pred_tw_no[i] == true_tw_no[i]:
                    tw_no_correct += 1
            position_accu = position_correct / len(pred_position) * 100
            armstand_accu = armstand_correct / len(pred_armstand) * 100
            rot_type_accu = rot_type_correct / len(pred_rot_type) * 100
            ss_no_accu = ss_no_correct / len(pred_ss_no) * 100
            tw_no_accu = tw_no_correct / len(pred_tw_no) * 100
            print('Accuracies: Position: ', position_accu, ' Armstand: ', armstand_accu, ' Rot_type: ', rot_type_accu,
                  ' SS_no: ', ss_no_accu, ' TW_no: ', tw_no_accu)

        if with_hockey_classification:
        # if with_caption:
            correct_class = {'switch_correct': 0, 'advance_correct': 0, 'faceoff_correct': 0, 'play_make_correct': 0,
                'play_receive_correct': 0, 'whistle_correct': 0, 'shot_correct': 0, 'hit_correct': 0, 'shot_block_correct': 0, 
                'penalty_correct': 0, 'ricochet_correct': 0}
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
                if pred_whistle[i] == true_whistle[i]:
                    correct_class['whistle_correct'] += 1
                if pred_shot[i] == true_shot[i]:
                    correct_class['shot_correct'] += 1
                if pred_hit[i] == true_hit[i]:
                    correct_class['hit_correct'] += 1
                if pred_shot_block[i] == true_shot_block[i]:
                    correct_class['shot_block_correct'] += 1
                if pred_penalty[i] == true_penalty[i]:
                    correct_class['penalty_correct'] += 1
                if pred_ricochet[i] == true_ricochet[i]:
                    correct_class['ricochet_correct'] += 1

            for k, v in correct_class.items():
                print('{}: {}'.format(k, v / len(pred_switch) * 100))

        if with_score_regression:
            rho, p = stats.spearmanr(pred_scores, true_scores)
            print('Predicted scores: ', pred_scores)
            print('True scores: ', true_scores)
            print('Correlation: ', rho)


def main():
    parameters_2_optimize = (list(model_CNN.parameters()) + list(model_my_fc6.parameters()))
    parameters_2_optimize_named = (list(model_CNN.named_parameters()) + list(model_my_fc6.named_parameters()))
    if with_score_regression:
        parameters_2_optimize = parameters_2_optimize + list(model_score_regressor.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_score_regressor.named_parameters())
    if with_dive_classification:
        parameters_2_optimize = parameters_2_optimize + list(model_dive_classifier.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_dive_classifier.named_parameters())
    if with_hockey_classification:
        parameters_2_optimize += list(model_hockey_classifier.parameters())
        parameters_2_optimize_named += list(model_hockey_classifier.named_parameters())
    if with_caption:
        parameters_2_optimize = parameters_2_optimize + list(model_caption.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_caption.named_parameters())

    optimizer = optim.Adam(parameters_2_optimize, lr = 0.0001)
    print('Parameters that will be learnt: ', 'parameters_2_optimize_named')

    criterions = {}
    if with_score_regression:
        criterion_final_score = nn.MSELoss()
        penalty_final_score = nn.L1Loss()
        criterions['criterion_final_score'] = criterion_final_score
        criterions['penalty_final_score'] = penalty_final_score
    if with_dive_classification:
        criterion_dive_classifier = nn.CrossEntropyLoss()
        criterions['criterion_dive_classifier'] = criterion_dive_classifier
    if with_hockey_classification:
        criterion_hockey_classifier = nn.CrossEntropyLoss()
        criterions['criterion_hockey_classifier'] = criterion_hockey_classifier
    if with_caption:
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

        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_my_fc6, 'model_my_fc6', epoch, saving_dir)
            if with_score_regression:
                save_model(model_score_regressor, 'model_score_regressor', epoch, saving_dir)
            if with_dive_classification:
                save_model(model_dive_classifier, 'model_dive_classifier', epoch, saving_dir)
            if with_hockey_classification:
                save_model(model_hockey_classifier, 'model_hockey_classifier', epoch, saving_dir)
            if with_caption:
                save_model(model_caption, 'model_caption', epoch, saving_dir)

    plt.plot(range(len(losses)), losses)
    plt.savefig('losses4.png')



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

    if with_score_regression:
        # loading our score regressor
        model_score_regressor = score_regressor()
        model_score_regressor = model_score_regressor.cuda()
        print('Using Final Score Loss')

    if with_dive_classification:
        # loading our dive classifier
        model_dive_classifier = dive_classifier()
        model_dive_classifier = model_dive_classifier.cuda()
        print('Using Dive Classification Loss')

    if with_hockey_classification:
        # loading our hockey classifier
        model_hockey_classifier = hockey_classifier()
        model_hockey_classifier = model_hockey_classifier.cuda()
        print('Using Hockey Classification Loss')

    if with_caption:
        # loading our caption model
        model_caption = S2VTModel(vocab_size, max_cap_len, caption_lstm_dim_hidden,
                                  caption_lstm_dim_word, caption_lstm_dim_vid,
                                  rnn_cell=caption_lstm_cell_type, n_layers=caption_lstm_num_layers,
                                  rnn_dropout_p=caption_lstm_dropout,
                                  eos_id=1719, sos_id=2060)
        model_caption = model_caption.cuda()
        print('Using Captioning Loss')

    main()