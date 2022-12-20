import time
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

from data_loader import TestData, process_sysu

gall_loader, query_loader = None, None
query_img, query_label, query_ca = [None] * 3
gall_img, gall_label, gall_cam = [None] * 3

img_h, img_w = 288, 144
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])

test_mode = [1, 1]
nquery, ngall = 0, 0

def load_data(args, test_batch=50, data_path='../Datasets/SYSU-MM01/', mode = 'Vis'):

    global gall_loader, query_loader, \
        nquery, ngall, query_img,\
        query_label, query_cam, gall_img, gall_label, gall_cam


    # testing set
    query_img, query_label, query_cam = process_sysu(data_path, data='query', mode = mode)
    gall_img, gall_label, gall_cam = process_sysu(data_path, data='gallery', mode = mode, single_shot=False)
    nquery = len(query_label)
    ngall = len(gall_label)

    gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(img_w, img_h))
    queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(img_w, img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=test_batch, shuffle=False, num_workers=args.workers)

    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))

def test(epoch, net):
    # switch to evaluation mode
    pool_dim = net.pool_dim
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, pool_dim))
    gall_feat_att = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, pool_dim))
    query_feat_att = np.zeros((nquery, pool_dim))
    time_inference = 0
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            start1 = time.time()
            feat, feat_att = net(input, input, modal=test_mode[1])
            time_inference += (time.time() - start1)
            #print('Extracting Time:\t {:.3f} len={:d}'.format(time.time() - start1, len(input)))

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time_inference))
    #exit(0)
    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    # if dataset == 'regdb':
    #     cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
    #     cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    # elif dataset == 'sysu':

    cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))


    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

def validate(epoch, model, args, mode = 'Vis'):
    global best_acc, best_epoch

    if gall_loader is None:
        load_data(args, mode=mode)

    cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch, model)
    # save model
    # if max(mAP, mAP_att) > best_acc:  # not the real best for sysu-mm01
    #     best_acc = max(mAP, mAP_att)
    #     best_epoch = epoch
    #     state = {
    #         'net': net.state_dict(),
    #         'cmc': cmc_att,
    #         'mAP': mAP_att,
    #         'mINP': mINP_att,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, checkpoint_path + suffix + '_best.t')

    # save model
    # if epoch > 10 and epoch % args.save_epoch == 0:
    #     state = {
    #         'net': net.state_dict(),
    #         'cmc': cmc,
    #         'mAP': mAP,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    id_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # order = indices[q_idx]
        # remove = (q_camid == 3) & (g_camids[order] == 2)
        # keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        id_cmc = pred_label[q_idx]  # [keep]
        id_cmc = id_cmc[np.sort(np.unique(id_cmc, return_index=True)[1])]

        new_match = (id_cmc == q_pid).astype(np.int32)
        id_cmc = new_match.cumsum()
        id_all_cmc.append(id_cmc[:max_rank])

        orig_cmc = matches[q_idx]  # [keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / (np.arange(len(orig_cmc)) + 1) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC

    id_all_cmc = np.asarray(id_all_cmc).astype(np.float32)
    id_all_cmc = id_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return id_all_cmc, mAP, mINP
