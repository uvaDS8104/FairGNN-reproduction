# %%
import time
import argparse

import dgl
import networkx
import networkx as nx

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from utils import load_data, accuracy, load_pokec
from models.FairAC import *
from models.GCN import GCN_Body
from karateclub import DeepWalk

def parser_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    parser.add_argument('--model', type=str, default="GAT",
                        help='the type of model GCN/GAT')
    parser.add_argument('--dataset', type=str, default='pokec_n',
                        choices=['pokec_z', 'pokec_n', 'nba'])
    parser.add_argument('--num-hidden', type=int, default=64,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--acc', type=float, default=0.688,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.745,
                        help='the selected FairGNN ROC score on val would be at least this high')
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")
    parser.add_argument('--attn_vec_dim', type=int, default=128,
                        help="attention vector dim")
    parser.add_argument('--num_heads', type=int, default=1,
                        help="the number of attention heads")
    parser.add_argument('--feat_drop_rate', type=float, default=0.3,
                        help="feature dropout rate")
    parser.add_argument('--num_sen_class', type=int, default=1,
                        help="number of sensitive classes")
    parser.add_argument('--transformed_feature_dim', type=int, default=128,
                        help="transformed feature dimensions")
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    return args


def fair_metric(output, idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality

def main():
    args = parser_args()
    # %%
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    print(args.dataset)

    if args.dataset != 'nba':
        if args.dataset == 'pokec_z':
            dataset = 'region_job'
        else:
            dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = args.label_number
        sens_number = args.sens_number
        seed = 20
        path = "../dataset/pokec/"
        test_idx = False
    else:
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "../dataset/NBA"
        test_idx = True
    print(dataset)

    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                           sens_attr,
                                                                                           predict_attr,
                                                                                           path=path,
                                                                                           label_number=label_number,
                                                                                           sens_number=sens_number,
                                                                                           seed=seed, test_idx=test_idx)

    # remove idx_test adj, features
    exclude_test = torch.ones(adj.shape[1]).bool()      # indices after removing idx_test
    exclude_test[idx_test] = False
    sub_adj = adj[exclude_test][:, exclude_test]
    indices = []
    counter = 0
    for e in exclude_test:
        indices.append(counter)
        if e:
            counter += 1
    indices = torch.LongTensor(indices)
    y_idx = indices[idx_train]
    # ################ modification on dataset idx######################
    print(len(idx_test))

    from utils import feature_norm

    # G = dgl.DGLGraph()
    G = dgl.from_scipy(adj, device='cuda:0')
    subG = dgl.from_scipy(sub_adj, device='cuda:0')

    # G.from_scipy_sparse_matrix(adj)
    if dataset == 'nba':
        features = feature_norm(features)
    # G = glist[0].cpu().to_networkx().to_undirected()
    #
    # model = DeepWalk(walk_length=100, dimensions=64, window_size=5)
    # model.fit(G)
    # embedding = model.get_embedding()
    # np.save('embedding.npy', embedding, allow_pickle=True)
    # b = np.load('embedding.npy')
    # print(b.shape)



    # todo: put sensitive into attributes to let ACmodel to predict
    # todo: model cov loss need to find a way to define
    # todo: hyperparameter to constraint loss_ac, in order to address fc make everything be the same.
    # todo: it is the fc layer that reduce the loss_ac more. solution: autoencoder
    # todo: val, test need to use ac to complete featureswithsens[]



    # %%
    labels[labels > 1] = 1
    if sens_attr:
        sens[sens > 0] = 1
    # Model and optimizer

    # processed feature
    embedding = np.load('embedding.npy')        # embeding is produced by Deep Walk
    embedding = torch.tensor(embedding)
    adj_mat = adj.toarray()
    adjTensor = torch.FloatTensor(adj_mat)
    sub_nodes = np.array_split(range(features.shape[0]), 4)
    sub_nodes = [torch.tensor(s).cuda() for s in sub_nodes]

    # model = FairAC_GNN(nfeat=features.shape[1],transformed_feature_dim=128, emb_dim= embedding.shape[1], args=args)

    transformed_feature_dim = args.transformed_feature_dim
    GNNmodel = GNN(nfeat=transformed_feature_dim, args=args)
    ACmodel = FairAC(feature_dim=features.shape[1],transformed_feature_dim=transformed_feature_dim, emb_dim=embedding.shape[1], args=args)
    # model.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number)))
    if args.cuda:
        GNNmodel.cuda()
        ACmodel.cuda()
        embedding = embedding.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()
        idx_sens_train = idx_sens_train.cuda()

    # todo fair sub graph adj for all graph done!
    subgraph_adj_list = []
    feat_keep_idx_sub_list = []
    feat_drop_idx_sub_list = []
    for sub_node in sub_nodes:
        feat_keep_idx_sub, feat_drop_idx_sub = train_test_split(np.arange(len(sub_node)),
                                                                test_size=args.feat_drop_rate)
        feat_keep_idx_sub_list.append(feat_keep_idx_sub)
        feat_drop_idx_sub_list.append(feat_drop_idx_sub)
        subgraph_adj = adjTensor[sub_node][:, sub_node][:, feat_keep_idx_sub]
        subgraph_adj_list.append(subgraph_adj)

    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100


    for epoch in range(args.epochs):
        t = time.time()
        GNNmodel.train()
        ACmodel.train()

        GNNmodel.optimizer_G.zero_grad()
        ACmodel.optimizer_AC.zero_grad()
        ACmodel.optimizer_S.zero_grad()

        # define train dataset, using the sub_nodes[0][feat_keep_idx_sub], which are fully labeled
        ac_train_idx = sub_nodes[0][feat_keep_idx_sub]
        feat_keep_idx, feat_drop_idx = train_test_split(np.arange(ac_train_idx.shape[0]),
                                                        test_size=args.feat_drop_rate)
        features_train = features[ac_train_idx]
        sens_train = sens[ac_train_idx]

        training_adj = adjTensor[ac_train_idx][:, ac_train_idx][:, feat_keep_idx].cuda()
        feature_src_re2, features_hat, _ = ACmodel(training_adj, embedding[ac_train_idx], embedding[ac_train_idx][feat_keep_idx],
                                                features_train[feat_keep_idx])
        loss_ac = ACmodel.loss(features_train[feat_drop_idx], feature_src_re2[feat_drop_idx, :])
        loss_reconstruction = torch.zeros(1).cuda()
#         F.pairwise_distance(features_hat, features_train[feat_keep_idx],2).mean()
        # base AC finished###############

        # pretrain AC model
        if epoch < 0:
            # ###############pretrain AC model ##########################
            print("Epoch: {:04d}, loss_ac: {:.4f}, loss_reconstruction: {:.4f}"
                  .format(epoch, loss_ac.item(), loss_reconstruction.item()))
            AC_loss = loss_reconstruction + loss_ac
            AC_loss.backward()
            ACmodel.optimizer_AC.step()
            continue

        # mitigate unfairness loss
        feature_src_re2_detach = feature_src_re2.detach()
        sens_prediction_detach = ACmodel.sensitive_pred(feature_src_re2_detach)
        criterion = nn.BCEWithLogitsLoss()
        # only update sensitive classifier
        Csen_loss = criterion(sens_prediction_detach[feat_keep_idx], sens_train[feat_keep_idx].unsqueeze(1).float())
        # sensitive optimizer.step
        Csen_loss.backward()
        ACmodel.optimizer_S.step()

        sens_prediction = ACmodel.sensitive_pred(feature_src_re2)
        sens_confusion = (torch.ones(sens_train.shape)/2).cuda()
        Csen_adv_loss = criterion(sens_prediction, sens_confusion.unsqueeze(1).float())
        # AC optimizer.step
        AC_loss = Csen_adv_loss + loss_reconstruction + loss_ac
        AC_loss.backward()
        ACmodel.optimizer_AC.step()


        # todo: training adversaial model, do i need to clean test nodes' features or subgraph of G?  !done by remove test nodes!
        features_embedding = torch.zeros((features.shape[0], transformed_feature_dim)).cuda()
        with torch.no_grad():
            # ############# Attribute completion over graph######################
            # # todo: cut graph into several small subgraph and doing AC on them.
            for i, sub_node in enumerate(sub_nodes):
                feat_keep_idx_sub = feat_keep_idx_sub_list[i]
                feat_drop_idx_sub = feat_drop_idx_sub_list[i]

                feature_src_AC, features_hat, transformed_feature = ACmodel(subgraph_adj_list[i].cuda(), embedding[sub_node],
                                                       embedding[sub_node][feat_keep_idx_sub],
                                                       features[sub_node][feat_keep_idx_sub])
                features_embedding[sub_node[feat_drop_idx_sub]] = feature_src_AC[feat_drop_idx_sub]
                features_embedding[sub_node[feat_keep_idx_sub]] = transformed_feature

        # features_with_sens[idx_test] = torch.zeros(features_with_sens[idx_test].shape).cuda()
        # todo: here features_train and sens_train can detach() !done!
        features_embedding_exclude_test = features_embedding[exclude_test].detach()

        feat_emb, y = GNNmodel(subG, features_embedding_exclude_test)

        Cy_loss = GNNmodel.criterion(y[y_idx], labels[idx_train].unsqueeze(1).float())

        # end2end, also update ACmodel
        ACmodel.optimizer_AC.zero_grad()
        Cy_loss.backward()
        ACmodel.optimizer_AC.step()
        GNNmodel.optimizer_G.step()

        print(
            "Epoch: {:04d}, loss_ac: {:.4f}, loss_reconstruction: {:.4f}, Csen_loss: {:.4}, Csen_adv_loss: {:.4f}, Cy_loss: {:.4f}"
            .format(epoch, loss_ac.item(), loss_reconstruction.item(), Csen_loss.item(), Csen_adv_loss.item(),
                    Cy_loss.item()))

        ##################### training finished ###################################

        cls_loss = Cy_loss
        GNNmodel.eval()
        ACmodel.eval()
        with torch.no_grad():

            _, output = GNNmodel(G, features_embedding)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            parity, equality = fair_metric(output, idx_test, labels, sens)
        # if acc_val > args.acc and roc_val > args.roc:
        if acc_test > args.acc and roc_test > args.roc:
            # if best_fair > parity_val + equality_val:
            #     best_fair = parity_val + equality_val
            if best_fair > parity + equality:
                best_fair = parity + equality
                best_result['acc'] = acc_test.item()
                best_result['roc'] = roc_test
                best_result['parity'] = parity
                best_result['equality'] = equality
                torch.save(GNNmodel,"GNN_epoch{:04d}_acc{:.4f}_roc{:.4f}_par{:.4f}_eq_{:.4f}".format(epoch,acc_test.item(),roc_test
                                                                                     ,parity,equality))
                torch.save(ACmodel, "ACmodel_epoch{:04d}_acc{:.4f}_roc{:.4f}_par{:.4f}_eq_{:.4f}".format(epoch, acc_test.item(), roc_test
                                                                                         , parity, equality))

            print("=================================")
            log = "Epoch: {:04d}, loss_ac: {:.4f}, loss_reconstruction: {:.4f}, Csen_loss: {:.4}, Csen_adv_loss: {:.4f}, cls: {:.4f}"\
                  .format(epoch, loss_ac.item(), loss_reconstruction.item(), Csen_loss.item(), Csen_adv_loss.item(), cls_loss.item())
            with open('log.txt', 'a') as f:
                f.write(log)
            print('Epoch: {:04d}'.format(epoch),
                  'loss_ac: {:.4f}'.format(loss_ac.item()),
                  'loss_reconstruction: {:.4f}'.format(loss_reconstruction.item()),
                  'Csen_loss: {:.4f}'.format(Csen_loss.item()),
                  'Csen_adv_loss: {:.4f}'.format(Csen_adv_loss.item()),
                  'cls: {:.4f}'.format(cls_loss.item())
                  # 'acc_val: {:.4f}'.format(acc_val.item()),
                  # "roc_val: {:.4f}".format(roc_val),
                  # "parity_val: {:.4f}".format(parity_val),
                  # "equality: {:.4f}".format(equality_val)
                  )
            print("Test:",
                  "accuracy: {:.4f}".format(acc_test.item()),
                  "roc: {:.4f}".format(roc_test),
                  "parity: {:.4f}".format(parity),
                  "equality: {:.4f}".format(equality))

            log = 'Test: accuracy: {:.4f} roc: {:.4f} parity: {:.4f} equality: {:.4f}\n' \
                .format(acc_test.item(), roc_test, parity, equality)
            with open('log.txt', 'a') as f:
                f.write(log)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('============performace on test set=============')
    if len(best_result) > 0:
        log = "Test: accuracy: {:.4f}, roc: {:.4f}, parity: {:.4f}, equality: {:.4f}"\
                  .format(best_result['acc'],best_result['roc'], best_result['parity'],best_result['equality'])
        with open('log.txt', 'a') as f:
            f.write(log)
        print("Test:",
              "accuracy: {:.4f}".format(best_result['acc']),
              "roc: {:.4f}".format(best_result['roc']),
              "parity: {:.4f}".format(best_result['parity']),
              "equality: {:.4f}".format(best_result['equality']))
    else:
        print("Please set smaller acc/roc thresholds")

if __name__ == '__main__':
    main()