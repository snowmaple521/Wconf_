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
import torch.nn.functional as F
import copy
import json
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
#caculate entroy
# def instance_bce_with_logits(logits, labels, reduction='mean'):
#     assert logits.dim() == 2
#     loss = F.binary_cross_entropy_with_logits(
#                                 logits, labels, reduction=reduction)
#     prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
# 
#     prediction_max, pred_ans_ind = torch.topk(F.softmax(logits, dim=-1), k=1, dim=-1, sorted=False)
#     neg_top_k = torch.gather(F.softmax(logits, dim=-1), 1, top_ans_ind).sum(1)
#     # pre_ans_k = pre_ans_k.tolist()
#     neg_top_k = neg_top_k.tolist()
#     pre_ans_k = prediction_max.squeeze(1)
#     # pred_ans_ind = pred_ans_ind.squeeze(1)
#     pre_ans_k = pre_ans_k.tolist()
#     # pred_ans_ind = pred_ans_ind.tolist()
# 
#     if reduction == "mean":
#         loss *= labels.size(1)
#     return loss,pre_ans_k,neg_top_k
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss
def calc_entropy(p):
    # size(att) = [b x g x v x q]
    eps = 1e-8
    return (-p * (p + eps).log()).sum(1) # g

def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        # result['entroy'] = entroy[i].item()
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results
def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]
def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model,train_loader, eval_loader,args,qid2type):

    dataset=args.dataset
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(model.parameters())

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    utils.print_model(model, logger)
    total_step = 0
    best_eval_score = 0



    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv #1
        top_hint=args.top_hint #9
        topq=args.topq#1
        keep_qtype=args.keep_qtype#true
        qvp=args.qvp #0



    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        results2 = []
        t = time.time()
        neg_list = []
        pred_max_list = []
        ans = {}
        for i, (v, q,a, b, hintscore,type_mask,q_mask,qids) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):

            total_step += 1
            #########################################
            # v = v['image_features']
            v = v.squeeze(1)
            v = v.float()
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q = q.float()
            q = q.squeeze(1)
            q_mask=Variable(q_mask).cuda() #[512,14] [18455,18455...,23,11,48]
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            # q = model_clip.encode_text(q)
            # qnoty = Variable(qnoty).cuda()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda() #[512,14] [0,0,0,...,1,1,1]
            # notype_mask=Variable(notype_mask).float().cuda() #[14,512] [0,0,0,...1,1,1]
            #########################################

            if mode=='updn':
                pred,loss,_ = model(v, q, a, b, None,None)
                # loss = instance_bce_with_logits(pred, a)
                # loss_r = instance_bce_with_logits(pred, a)
                loss_self,pred_ansind,pred_max = compute_self_loss(pred, a)
                # print("loss_self:",loss_self)
                neg_list = neg_list+pred_ansind
                pred_max_list = pred_max_list+pred_max
                # loss = loss_r
                # print("loss_r:",loss_r)
                loss = loss + args.a * loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

            elif mode=='q_debias':

                sen_mask=type_mask

                ## first train
                pred, loss_r,word_emb = model(v, q, a, b, None,None)

                word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), word_emb, create_graph=True)[0]
                loss_self, _, _ = compute_self_loss(pred, a)
                loss = loss_r + args.a * loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ## second train

                word_grad_cam = word_grad.sum(2)
                # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask

                w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]
                # q_mask = 1
                q2 = copy.deepcopy(q_mask)

                m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                if dataset=='cpv1':
                    m3=m1*18330
                else:
                    m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                q2 = q2 * m2.long() + m3.long()

                pred, _, _ = model(v, q2, None, b, None,epoch)

                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)
                a2 = a * false_ans
                q3 = copy.deepcopy(q)
                if dataset=='cpv1':
                    q3.scatter_(1, w_ind, 18330)
                else:
                    q3.scatter_(1, w_ind, 18455)

                # third train

                pred, loss, _ = model(v, q3, a2, b, None,None)
                # loss_self,_,_ = compute_self_loss(pred,a2)
                # loss = loss_r+3.4*loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)

            elif mode=='v_debias':
                ## first train
                pred, loss_r, _ = model(v, q, a, b, None,None)
                visual_grad=torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]
                loss_self,_,_ = compute_self_loss(pred,a)
                loss = loss_r + 3.4*loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ##second train
                v_mask = torch.zeros(v.shape[0], 36).cuda()
                visual_grad_cam = visual_grad.sum(2)
                hint_sort, hint_ind = hintscore.sort(1, descending=True)
                v_ind = hint_ind[:, :top_hint]
                v_grad = visual_grad_cam.gather(1, v_ind)

                if topv==-1:
                    v_grad_score,v_grad_ind=v_grad.sort(1,descending=True)
                    v_grad_score=nn.functional.softmax(v_grad_score*10,dim=1)
                    v_grad_sum=torch.cumsum(v_grad_score,dim=1)
                    v_grad_mask=(v_grad_sum<=0.65).long()
                    v_grad_mask[:,0] = 1
                    v_mask_ind=v_grad_mask*v_ind
                    for x in range(a.shape[0]):
                        num=len(torch.nonzero(v_grad_mask[x]))
                        v_mask[x].scatter_(0,v_mask_ind[x,:num],1)
                else:
                    v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
                    v_star = v_ind.gather(1, v_grad_ind)
                    v_mask.scatter_(1, v_star, 1)

                #2 train
                pred, _, _ = model(v, q, None, b, v_mask,None)

                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)
                a2 = a * false_ans

                v_mask = 1 - v_mask
                #3 train
                pred, loss, _ = model(v, q, a2, b, v_mask,None)
                # loss_self, _, _ = compute_self_loss(pred, a2)
                # loss = loss_r + 3.4 * loss_self
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)

            elif mode=='q_v_debias':
                random_num = random.randint(1, 10)

                sen_mask = type_mask
                if random_num<=qvp: #qss
                    ## first train
                    pred1, loss_r, word_emb = model(v, q, a, b, None,epoch)
                    word_grad = torch.autograd.grad((pred1 * (a > 0).float()).sum(), word_emb, create_graph=True)[0]
                    loss_self,_,_ = compute_self_loss(pred1,a)
                    loss = args.a*loss_self+loss_r
                    # loss = loss_r
                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred1, a.data).sum()
                    train_score += batch_score

                    ## second train

                    word_grad_cam = word_grad.sum(2)
                    # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                    word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                    word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask
                    w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]
                    # q_mask = 1
                    q2 = copy.deepcopy(q_mask)

                    m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                    m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                    m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                    if dataset=='cpv1':
                        m3=m1*18330
                    else:
                        m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                    q2 = q2 * m2.long() + m3.long()

                    pred2, _, _ = model(v, q2, None, b, None,epoch)

                    pred_ind = torch.argsort(pred2, 1, descending=True)[:, :5]
                    false_ans = torch.ones(pred2.shape[0], pred2.shape[1]).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans
                    q3 = copy.deepcopy(q)
                    if dataset=='cpv1':
                        q3.scatter_(1, w_ind, 18330)
                    else:
                        q3.scatter_(1, w_ind, 18455)

                    ## third train
                    # pred1 = Variable(pred1).cuda().requires_grad_()
                    pred3, loss_r, _ = model(v, q3, a2, b, None,epoch)
                    # loss_self,_,_ = compute_self_loss(pred3,a2)
                    # loss = loss_r+3.4*loss_self
                    loss = loss_r
                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)


                else: # #vss
                    ## first train a:[512,2274] b:[512,2274] q;[512,14] v[512,36,2048]
                    pred4, loss_r, _ = model(v, q, a, b, None,epoch) #pred[512,2274] loss=tensor(669.1957, _:[512,14,300]
                    #torch.Size([512, 36, 2048])
                    # print((pred4 * (a > 0).float()).sum())
                    visual_grad = torch.autograd.grad((pred4 * (a > 0).float()).sum(), v, create_graph=True)[0]

                    loss_self, _, _ = compute_self_loss(pred4, a)
                    loss = loss_r + 3.4 * loss_self
                    # loss = loss_r
                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred4, a.data).sum()
                    train_score += batch_score

                    ##second train
                    v_mask = torch.zeros(v.shape[0], 36).cuda() #[512,36] all 0
                    visual_grad_cam = visual_grad.sum(2) #[512,36]
                    #hint_sort:512,36
                    hint_sort, hint_ind = hintscore.sort(1, descending=True)
                    v_ind = hint_ind[:, :top_hint]
                    v_grad = visual_grad_cam.gather(1, v_ind)
                    # pred4 = Variable(pred4).cuda().requires_grad_()
                    if topv == -1:
                        v_grad_score, v_grad_ind = v_grad.sort(1, descending=True)
                        v_grad_score = nn.functional.softmax(v_grad_score * 10, dim=1)
                        v_grad_sum = torch.cumsum(v_grad_score, dim=1)
                        v_grad_mask = (v_grad_sum <= 0.65).long()
                        v_grad_mask[:,0] = 1
                        v_mask_ind = v_grad_mask * v_ind
                        for x in range(a.shape[0]):
                            num = len(torch.nonzero(v_grad_mask[x]))
                            v_mask[x].scatter_(0, v_mask_ind[x,:num], 1)
                    else:
                        v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
                        v_star = v_ind.gather(1, v_grad_ind)
                        v_mask.scatter_(1, v_star, 1)

                    #2 train answer=None,
                    pred5, _, _ = model(v, q, None, b, v_mask,epoch)
                    pred_ind = torch.argsort(pred5, 1, descending=True)[:, :5]
                    false_ans = torch.ones(pred5.shape[0], pred5.shape[1]).cuda() #[512,2274] all 1
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans #a-

                    v_mask = 1 - v_mask  #512,36 all:1

                    #3 train
                    pred6, loss_r, _ = model(v, q, a2, b, v_mask,None)
                    # loss_self, _, _ = compute_self_loss(pred6, a2)
                    loss = loss_r
                    # loss_self = compute_self_loss(pred6, a2)
                    # loss = loss_r + 3.4 * loss_self
                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
        # ans['pred_max'] = pred_max_list
        # ans['ans_indx_pred'] = neg_list
        # json_str = json.dumps(ans, indent=4)  # import torch
        # with open('..\\resutls\\answer_train_conf2-{}.json'.format(epoch), 'w') as json_file:  # import numpy
        #     json_file.write(json_str)  # a = torch.rand(4,1)
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
            eval_loss = results['total_loss']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\tbest eval score: %.2f (%.2f)' % (100 * best_eval_score, 100 * bound))
            logger.write('\teval_loss: %.2f, score: %.2f(%.2f)' % (eval_loss, eval_score,100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score




def evaluate(model, dataloader, qid2type,epoch):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_loss = 0
    total_number = 0
    total_other = 0
    results2 = []
    for v, q, a, b, _,_,qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        v = v.squeeze(1)
        v = v.float()
        q = q.float()
        q = q.squeeze(1)
        pred,_,_= model(v, q, None, None, None,None)
        loss = instance_bce_with_logits(pred.cpu(), a)
        # loss_self, _, _ = compute_self_loss(pred.cpu(), a)
        # loss = loss_r + 2.4 * loss_self
        # pred_sigmod = torch.sigmoid(pred)
        # entroy = calc_entropy(pred_sigmod)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        total_loss += loss.item() * q.size(0)
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        qid = qids
        pred = pred.cpu()
        current_results = make_json(pred, qid, dataloader)
        results2.extend(current_results)


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

    score = score / len(dataloader.dataset)
    total_loss = total_loss / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    # dir_exp = 'data/'
    # dir_rslt = os.path.join(dir_exp, 'css_clip_all_vqa_k1', 'test', 'epoch,{}'.format(epoch))
    # os.makedirs(dir_rslt)
    # path_rslt = os.path.join(dir_rslt, 'cp_v2_{}_model_results.json'.format('test2014'))
    # with open(path_rslt, 'w') as f:
    #     json.dump(results2, f)
    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
        total_loss = total_loss
    )
    return results
