import torch
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F
from dhg.models import HGNN, HGNNP, GAT, GCN

class SCELoss(nn.Module):

    def __init__(self, alpha:float=3):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)
        loss = loss.mean()
        return loss
    
def setup_module(m_type, in_dim, hid_dim, out_dim, use_bn, dropout) -> nn.Module:
    if m_type == "hgnn":
        mod = HGNN(
            in_channels=in_dim,
            hid_channels=hid_dim,
            num_classes=out_dim,
            use_bn=use_bn,
            drop_rate=dropout,
        )
    elif m_type == "hgnnp":
        mod = HGNNP(
            in_channels=in_dim,
            hid_channels=hid_dim,
            num_classes=out_dim,
            use_bn=use_bn,
            drop_rate=dropout,
        )
    else:
        raise NotImplementedError
    return mod

def drop_features(x, p):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t())
        return sim

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, op_sim, use_sim):
    f = lambda x: torch.exp(x / T)
    sim1_1 =sim(z1,z1)
    sim1_2 =sim(z1,z2)
    refl_sim = f(sim1_1)
    between_sim = f(sim1_2)
    if use_sim:
        loss = -op_sim * torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    else:
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    return loss

def contrastive_loss(x1, x2, op_sim, use_sim=True):
    T = 0.5
    l1 = semi_loss(x1, x2, T, op_sim, use_sim)
    l2 = semi_loss(x2, x1, T, op_sim, use_sim)
    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    
    return ret
    
class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            edge_dim:int,
            feat_drop: float,
            use_bn: bool = False,
            mask_rate: float = 0.5,
            encoder_type: str = "hgnn",
            decoder_type: str = "hgnn",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = hid_dim
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.iso_head = nn.Sequential(nn.Linear(hid_dim*2,hid_dim*2),
                                      nn.Linear(hid_dim*2,1),
                                      nn.ReLU(inplace=True),
                                      nn.Sigmoid())
        self.sim_proj = nn.Sequential(nn.Linear(edge_dim,64),
                                      nn.Linear(64,edge_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Sigmoid()
                                      )        
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            use_bn=use_bn,
            dropout=feat_drop,
        )
        # build decoder
        self.decoder = setup_module(
            m_type=decoder_type,
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=in_dim,
            use_bn=use_bn,
            dropout=feat_drop,
        )
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Linear(hid_dim, hid_dim, bias=False)
        self.fc1_n = nn.Linear(hid_dim, hid_dim)
        self.fc2_n = nn.Linear(hid_dim, hid_dim)
        self.fc1_e = nn.Linear(hid_dim, hid_dim)
        self.fc2_e = nn.Linear(hid_dim, hid_dim)

        #loss function
        self.criterion = self.setup_loss_fn(loss_fn)


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    
    def node_projection(self, z):
        return self.fc2_n(F.elu(self.fc1_n(z)))
    
    def edge_projection(self, z):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    #loss func
    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = SCELoss()
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, x, hg, mask_rate):
        num_nodes = len(x)
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_hg = hg.clone()
        return use_hg, out_x, (mask_nodes, keep_nodes)

    def forward_attr(self, x, hg):

        loss = self.mask_attr_prediction(x, hg)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, x, hg):
        use_hg, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, hg, self._mask_rate)
        enc_rep = self.encoder(use_x, use_hg)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        #---- re-mask ----
        rep[mask_nodes] = 0
        recon = self.decoder(rep, use_hg)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss

    def forward_cl(self, x, hg1, hg2, op_sim, use_sim):
        loss = self.hyperedge_structure_loss(x, hg1, hg2, op_sim, use_sim)
        return loss
        
    def hyperedge_structure_loss(self, x, hg1, hg2, op_sim, use_sim):
        x1 = drop_features(x, 0.4)
        x_aug1 = self.encoder(x1,hg1)
        xe_aug1 = self.edge_projection(hg1.v2e(x_aug1))
        x2 = drop_features(x, 0.4)
        x_aug2 = self.encoder(x2,hg2)
        xe_aug2 = self.edge_projection(hg2.v2e(x_aug2))
        loss = contrastive_loss(xe_aug1, xe_aug2, op_sim, use_sim)
        return loss

    def embed(self, x, hg):
        rep = self.encoder(x, hg)
        return rep


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    

    
class MLP_classifier(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear1 = nn.Linear(num_dim, 256)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(256, num_class)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, hg, *args):
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits
    