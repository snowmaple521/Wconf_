import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy
import torch.nn.functional as F
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
                                logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss
# def compute_score_with_logits(logits, labels):
#     # logits = torch.max(logits, 1)[1].data # argmax
#     prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
#     neg_top_k = torch.gather(F.softmax(logits, dim=-1), 1, top_ans_ind).sum(1)
#     prediction_max, pred_ans_ind = torch.topk(F.softmax(logits, dim=-1), k=1, dim=-1, sorted=False)
#     pre_ans_k = prediction_max.squeeze(1)
#     logits = torch.argmax(logits,1)
#     pre_ans_k = pre_ans_k.tolist()
#     neg_top_k = neg_top_k.tolist()
#     one_hots = torch.zeros(*labels.size()).cuda()
#     one_hots.scatter_(1, logits.view(-1, 1), 1)
#     scores = (one_hots * labels)
#     return scores,pre_ans_k,neg_top_k

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    prediction_max, pred_ans_ind = torch.topk(F.softmax(logits_neg, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    pre_ans_k = prediction_max.squeeze(1)
    # neg_top_k = neg_top_k.squeeze(1)
    qice_loss = neg_top_k.mean()
    pre_ans_k = pre_ans_k.tolist()
    neg_top_k = neg_top_k.tolist()
    return qice_loss,pre_ans_k,neg_top_k
def train(model, train_loader, eval_loader,args,qid2type):
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0



    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp



    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        results2 = []
        pred_right_list = []
        pred_max_list = []
        ans = {}
        for i, (v, q,a, b, hintscore,type_mask,q_mask,qids) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1
            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            # q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            # q_noty = Variable(q_noty).cuda()
            b = Variable(b).cuda()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda()
            # notype_mask=Variable(notype_mask).float().cuda()
            #########################################

            if mode=='updn':
                pred,loss,_ = model(v, q, a, b, None)
                # loss = instance_bce_with_logits(pred,a)


                # loss_r = instance_bce_with_logits(pred, a)
                loss_self,_,_= compute_self_loss(pred, a)
                loss = loss+1.8*loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                # batch_score, pre_max, pred_right = compute_score_with_logits(pred, a.cuda())
                # batch_score = batch_score.cpu().numpy().sum(1)
                # pred_right_list = pred_right_list + pred_right
                # pred_max_list = pred_max_list + pre_max
                train_score += batch_score
                # train_score += batch_score.sum()

        if mode=='updn':
            total_loss /= len(train_loader.dataset)
        else:
            total_loss /= len(train_loader.dataset) * 2
        train_score = 100 * train_score / len(train_loader.dataset)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type,epoch)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            eval_loss_rubi = results['total_loss_rubi']
            eval_loss_bce = results['total_loss_bce']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\teval_loss_rubi: %.2f,eval_loss_bce: %.2f, score: %.2f(%.2f)' % (
            eval_loss_rubi, eval_loss_bce, eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))
        # ans['pred_max'] = pred_max_list
        # ans['pred_right'] = pred_right_list
        # json_str = json.dumps(ans, indent=4)  # import torch
        # with open('/home/admin888/tmp/pycharm_project_css/logs/results_rubi_conf/rubi_train_conf314_answer-{}.json'.format(epoch), 'w') as json_file:  # import numpy
        #     json_file.write(json_str)  # a = torch.rand(4,1)

        # ans['pred_max'] = pred_max_list
        # ans['pred_right'] = pred_right_list
        # json_str = json.dumps(ans, indent=4)  # import torch
        # with open(
        #         '/home/admin888/tmp/pycharm_project_css/logs/results_updn_conf/updn_train_conf321_answer-{}.json'.format(
        #                 epoch), 'w') as json_file:  # import numpy
        #     json_file.write(json_str)
        if eval_score > best_eval_score:
                    model_path = os.path.join(output, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                    best_eval_score = eval_score

                        # a = torch.rand(4,1)



def evaluate(model, dataloader, qid2type,epoch):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    total_loss_rubi = 0
    total_loss_bce = 0
    pred_right_list = []
    pred_max_list = []
    ans = {}

    for v, q, a, b, _,_,qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        # pred,loss_rubi,_ = model(v, q, a, None,None)
        pred, loss_rubi, _ = model(v, q, a, None, None)
        loss_bce = instance_bce_with_logits(pred.cpu(), a.cpu())
        total_loss_bce += loss_bce.item() * q.size(0)
        total_loss_rubi += loss_rubi.item() * q.size(0)
        # batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        # batch_score = compute_score_with_logits(pred, a.cuda())
        # batch_score = batch_score.cpu().numpy().sum(1)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        # pred_right_list = pred_right_list + pred_right
        # pred_max_list = pred_max_list + pre_max
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    # ans['pred_max'] = pred_max_list
    # ans['pred_right'] = pred_right_list
    # json_str = json.dumps(ans, indent=4)  # import torch
    # with open('rubi_test_yuanshi_answer-{}.json'.format(epoch), 'w') as json_file:  # import numpy
    #     json_file.write(json_str)
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    total_loss_rubi = total_loss_rubi / len(dataloader.dataset)
    total_loss_bce = total_loss_bce / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
        total_loss_rubi=total_loss_rubi,
        total_loss_bce=total_loss_bce
    )
    return results
