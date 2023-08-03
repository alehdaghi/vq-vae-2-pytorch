import time
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from scipy.spatial.distance import cdist

from data_loader import TestData, process_sysu
from torchvision import utils

from part.criterion import contrastive_loss

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
invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])



nquery, ngall = 0, 0

def load_data(args, test_batch=50, data='query', mode='all', single=True, data_path='../Datasets/SYSU-MM01/'):
    imgs, labels, cams = process_sysu(data_path, data=data, mode = mode, single_shot=single)
    n = len(imgs)
    dataset = TestData(imgs, labels, cams, transform=transform_test, img_size=(img_w, img_h))
    loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch, shuffle=False, num_workers=0)
    print('  {}    | {:5d} | {:8d}'.format(data, len(np.unique(labels)), n))
    return loader, n, imgs, labels, cams

def ext_feat(net, loader, modal=0):
    pool_dim = net.pool_dim
    net.eval()
    print('Extracting Feature...')
    start = time.time()
    ptr = 0
#     feat = np.zeros((ngall, pool_dim))
    n = len(loader.dataset)
    feats = np.zeros((n, pool_dim))
    labels = np.zeros(n)
    cams = np.zeros(n)
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att = net(input, input, modal=modal)
            feats[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()[:, :pool_dim]
            labels[ptr:ptr+batch_num] = label
            cams[ptr:ptr + batch_num] = cam
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return feats, labels, cams

dist_type='cosine'
def test(epoch, net, gall_loader, query_loader, test_mode = [1, 2]):
    # switch to evaluation mode
    gall_feat_att, g_l, gall_cam = ext_feat(net, gall_loader, modal=test_mode[0])
    # print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    query_feat_att, q_l, query_cam = ext_feat(net, query_loader, modal=test_mode[1])
    # switch to evaluation
    if (dist_type == 'cosine'):
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    else:
        distmat_att = -calc_dist(query_feat_att, gall_feat_att)

    cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, q_l, g_l, query_cam, gall_cam)
    # print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc_att, mAP_att, mINP_att

def testAll(epoch, net, gall_loader, query_loader, test_mode = [1, 2]):
    query_feat_att, q_l, query_cam = ext_feat(net, query_loader, modal=test_mode[1])

    cmc, mAP, mINP = None, None, None
    for i in range(10):
        gall_loader, ngall, _, _, _ = load_data(args=None, data='gallery', mode='all', single=True)
        gall_feat_att, g_l, gall_cam = ext_feat(net, gall_loader, modal=test_mode[0])
        # switch to evaluation
        if (dist_type == 'cosine'):
            distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
        else:
            distmat_att = -calc_dist(query_feat_att, gall_feat_att)

        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, q_l, g_l, query_cam, gall_cam)
        if cmc is None :
            cmc, mAP, mINP = cmc_att, mAP_att, mINP_att
        else:
            cmc += cmc_att
            mAP += mAP_att
            mINP += mINP_att
        print('iter:{}:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format( i+1,
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    cmc /=10
    mAP /= 10
    mINP /=10
    print('All:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    return cmc, mAP, mINP


def validate(epoch, net, args, mode = 'Vis', randGallery=False):

    global best_acc, best_epoch
    global gall_loader, query_loader, nquery, ngall

    if gall_loader is None or randGallery is True:
        gall_loader, ngall, _, _, _ = load_data(args, data='gallery', mode=mode, single=False)

    if query_loader is None:
        query_loader, nquery, _, _, _ = load_data(args, data='query', mode=mode)

    if mode == 'Vis':
        test_mode = [1, 1]
    elif mode == 'Ir':
        test_mode = [2, 2]
    else:
        test_mode = [1, 2]

    cmc_att, mAP_att, mINP_att = test(epoch, net, gall_loader, query_loader, test_mode)
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

    # print(
    #     'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    #         cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    return mAP_att


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


def calc_dist(probFea, galFea):
    k1, k2 = 20, 6
    lambda_value = 0.3
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = np.append(probFea, galFea, axis=0)
    feat = feat.astype(np.float16)
    # print('computing original distance')
    original_dist = cdist(feat, feat).astype(np.float16)
    original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

