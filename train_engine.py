import torch
import logging
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch import optim as optim
from model import MLP_classifier
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix


def pretrain_fmri(model, preprocessed_data, HG_aug_list, optimizer, scheduler, max_epoch, device, topo_sim_all, cl, attr, use_sim,logger):
    """
    input:
        model: model
        preprocessed_data: data list
        HG_aug_list: list of (HG_aug1, HG_aug2)
        optimizer: optimizer
        scheduler: learning rate scheduler
        max_epoch: int
        device: torch.device
        topo_sim_all: list of topo_sim
        cl: bool
        attr: bool
        use_sim: bool
        
    """
    if not cl:
        logger.info("Without Contrastive Learning")
    if not attr:
        logger.info("Without Attribute Learning")
    
    preprocessed_data = preprocessed_data
    HG_aug_list = [(HG_aug1.to(device), HG_aug2.to(device)) for HG_aug1, HG_aug2 in HG_aug_list]
    
    epoch_iter = range(max_epoch)
    best_loss = 1000    
    best_model = None
    last_model = None
    for epoch in epoch_iter:
        logger.info(f"---- Epoch {epoch+1}/{max_epoch} ----")
        
        losses = []
        for i, (HG_aug1, HG_aug2) in enumerate(tqdm(HG_aug_list)):
            model.train()
            x, hg, y = preprocessed_data[i]
            topo_sim = topo_sim_all[i]
            x = x.to(device)
            hg = hg.to(device)
            y = y.to(device)
            HG_aug1 = HG_aug1.to(device)
            HG_aug2 = HG_aug2.to(device)
            topo_sim = topo_sim.to(device)
            if cl:
                loss_cl = model.forward_cl(x, HG_aug1, HG_aug2, topo_sim, use_sim)  # topo
            else:
                loss_cl = 0
            if attr:
                loss_attr, _ = model.forward_attr(x, hg)    # seman
            else:
                loss_attr = 0
            loss = loss_attr * attr + loss_cl * cl
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_epoch = sum(losses) / len(losses)
        logger.info(f"# Epoch {epoch+1}: train_loss: {loss_epoch:.4f}")
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_model = deepcopy(model)
        last_model = deepcopy(model)
        
    return best_model, last_model



def compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity



def graph_classification_evaluation(k, model, preprocessed_data, num_classes, 
                                    lr_f, weight_decay_f, max_epoch_f, device, 
                                    train_mask, val_mask, test_mask, lbl, logger, linear_prob=True):
    model = model.to(device)
    model.eval()
    hg_emb_list = []

    with torch.no_grad():
        for i, (x, hg, y) in enumerate(preprocessed_data):
            x = x.to(device)
            hg = hg.to(device)
            node_embs = model.embed(x, hg)  
            graph_emb = node_embs.view(-1)
            graph_emb = graph_emb.unsqueeze(0) 
            hg_emb_list.append(graph_emb)
            
        
        all_hg_emb = torch.cat(hg_emb_list, dim=0)  

    labels = lbl.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    metric_dict = MLP_tune(
        all_hg_emb, max_epoch_f, num_classes, lr_f, weight_decay_f,
        device, train_mask, val_mask, test_mask, labels, logger,mute=True
    )
    return metric_dict


def MLP_tune(
        all_hg_emb, max_epoch, num_classes, lr_f, weight_decay_f,
        device, train_mask, val_mask, test_mask, labels,logger,  mute=False):

    all_hg_emb = all_hg_emb.to(device)
    labels = labels.to(device)
    labels = labels.long()
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None
    best_val_recall = 0
    best_val_precision = 0
    best_val_f1 = 0
    best_val_auc = 0
    best_val_specificity = 0
    
    

    classifier = MLP_classifier(all_hg_emb.shape[1], num_classes)   
    num_finetune_params = [p.numel() for p in classifier.parameters() if p.requires_grad]
    logger.info(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    classifier.to(device)
    optimizer_f = create_optimizer("adam", classifier, lr_f, weight_decay_f)

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        classifier.train()
        out = classifier(all_hg_emb[train_mask], None)
        loss = F.cross_entropy(out, labels[train_mask])
        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()

        with torch.no_grad():
            classifier.eval()
            pred = classifier(all_hg_emb, None) 
            pred_labels = pred.argmax(dim=1)
            
            val_pred = pred[val_mask]
            val_true = labels[val_mask]
            val_pred_labels = val_pred.argmax(dim=1)
        
            
            val_acc = accuracy_score(val_true.cpu(), val_pred_labels.cpu())
            val_loss = F.cross_entropy(val_pred, val_true)
            val_recall = recall_score(val_true.cpu(), val_pred_labels.cpu(), average='macro', zero_division=0)
            val_precision = precision_score(val_true.cpu(), val_pred_labels.cpu(), average='macro', zero_division=0)
            val_f1 = f1_score(val_true.cpu(), val_pred_labels.cpu(), average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(val_true.cpu(), F.softmax(val_pred, dim=1)[:, 1].cpu(),multi_class='ovr')   # multi_class='ovr' 时，roc_auc_score支持多类
            except ValueError:
                val_auc = 0.0  
            val_specificity = compute_specificity(val_true.cpu(), val_pred_labels.cpu())


        epoch_iter.set_description(
            f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, "
            f"val_acc:{val_acc:.4f}, val_recall:{val_recall:.4f}, val_precision:{val_precision:.4f}, "
            f"val_f1:{val_f1:.4f}, val_auc:{val_auc:.4f}"
        )
        
        # Set accuracy as the main metric to select the best model. Also can use other metrics like F1, AUC, etc. as the main metric.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = deepcopy(classifier)
            best_val_recall = val_recall 
            best_val_precision = val_precision
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_val_specificity = val_specificity
        elif val_acc == best_val_acc:
            best_val_recall = max(val_recall, best_val_recall)
            best_val_precision = max(val_precision, best_val_precision)
            best_val_f1 = max(val_f1, best_val_f1)
            best_val_auc = max(val_auc, best_val_auc)
            best_val_specificity = max(val_specificity, best_val_specificity)
            
    
    logger.info(f"--- Best Val Results ---")
    logger.info(f"Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Val Recall (Sensitivity): {best_val_recall:.4f}")
    logger.info(f"Val Specificity: {best_val_specificity:.4f}")
    logger.info(f"Val Precision: {best_val_precision:.4f}")
    logger.info(f"Val F1 Score: {best_val_f1:.4f}")
    logger.info(f"Val AUC: {best_val_auc:.4f}")    
    logger.info(f"Best epoch {best_val_epoch}")
            
    return {
        "test_acc": best_val_acc,
        "test_recall": best_val_recall,
        "test_precision": best_val_precision,
        "test_f1": best_val_f1,
        "test_auc": best_val_auc,
        "test_specificity": best_val_specificity,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch
        
    }
            

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer

