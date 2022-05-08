import argparse
import json
import sys
import torch
from torch.utils.data import DataLoader
import os
# from new_dataset import Dictionary, VQAFeatureDataset
from dataset import Dictionary, VQAFeatureDataset
import base_model
# import base_model
import utils
from vqa_debias_loss_functions import *
from tqdm import tqdm
from torch.autograd import Variable
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]
def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=True,
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2', help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--debias', default="none",
        choices=["learned_mixin", "reweight", "bias_product", "none"],
        help="Kind of ensemble loss to use")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--model_state', type=str, default='/home/admin888/tmp/pycharm_project_css/logs/logs/updn/updn_conf_zuixin_a_18_315_cpv1/model.pth')
    # parser.add_argument('--model_state', type=str, default='/home/admin888/tmp/pycharm_project_css/logs/logs/css_updn_vqacpself_loss_besttt/model.pth')
    args = parser.parse_args()
    return args
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
sys.stdout = Logger('-updn-conf22-.txt')
def compute_score_with_logits2(logits, labels):
    logits = torch.argmax(logits,1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
def compute_score_with_logits(logits, labels):
    # logits = torch.max(logits, 1)[1].data # argmax
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits, dim=-1), 1, top_ans_ind).sum(1)
    prediction_max, pred_ans_ind = torch.topk(F.softmax(logits, dim=-1), k=1, dim=-1, sorted=False)
    pre_ans_k = prediction_max.squeeze(1)
    logits = torch.argmax(logits,1)
    pre_ans_k = pre_ans_k.tolist()
    neg_top_k = neg_top_k.tolist()
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores,pre_ans_k,neg_top_k
def calc_entropy(p):
    # size(att) = [b x g x v x q]
    eps = 1e-8
    return (-p * (p + eps).log()).sum(1) # g

def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results
def evaluate(model,dataloader,qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    results = []
    model.train(False)
    pred_right_list = []
    pred_max_list = []
    ans = {}
    for v, q, a, b,_,_,qids,in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        pred, _,_= model(v, q, None, None,None,None)
        _, pre_max, pred_right = compute_score_with_logits(pred, a.cuda())
        batch_score = compute_score_with_logits2(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        pred_right_list = pred_right_list + pred_right
        pred_max_list = pred_max_list + pre_max
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid=qids[j]
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
        results.append(make_json(pred, qids, dataloader))
    # with open("rubi_conf_results.json", 'w') as f:
    #     json.dump(results, f)
    ans['pred_max'] = pred_max_list
    ans['pred_right'] = pred_right_list
    json_str = json.dumps(ans, indent=4)  # import torch
    with open('/home/admin888/tmp/pycharm_project_css/logs/updn_test_conf321_answer.json', 'w') as json_file:  # import numpy
        json_file.write(json_str)  # a = torch.rand(4,1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    print('\teval overall score: %.2f' % (100 * score))
    print('\teval up_bound score: %.2f' % (100 * upper_bound))
    print('\teval y/n score: %.2f' % (100 * score_yesno))
    print('\teval other score: %.2f' % (100 * score_other))
    print('\teval number score: %.2f' % (100 * score_number))


def evaluate_ai(model, dataloader):

    for v, q, a, b,_,_,qids, in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda().float().requires_grad_()
        q = Variable(q, requires_grad=False).cuda()
        a = a.cuda()
        # hintscore = hintscore.cuda().float()
        pred, _, _ = model(v, q, None, None, None, None)
        # print(pred[:, :3])
        vqa_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]  # [b , 36, 2048]
        vqa_grad_cam = vqa_grad.sum(2)  # (b, 36)
        sv_ind = torch.argmax(vqa_grad_cam, 1)  # b max_weizhi
        sv_ind = sv_ind.cpu().numpy()
        qids = qids.cpu().numpy()

        print(qids, sv_ind)
def main():
    args = parse_args()
    dataset = args.dataset


    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
    #                               cache_image_features=args.cache_features)
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)
    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()

    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    else:
        raise RuntimeError(args.mode)


    model_state = torch.load(args.model_state)
    model.load_state_dict(model_state, False)
    model = model.cuda()

    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # The original version uses multiple workers, but that just seems slower on my setup
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)
    print("Starting eval...")
    evaluate(model,eval_loader,qid2type)
    # evaluate_ai(model,eval_loader)
if __name__ == '__main__':
    main()
