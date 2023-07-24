import torch
import torch.nn as nn
from torch.nn.init import constant_
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, name, params):
        super(MLP, self).__init__()
        self.model_name = name
        self.params = params
        self.dataset_source = params["dataset_source"]
        if params["dataset_source"] == "criteo":
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(55824, 64),
                nn.Embedding(5443, 64),
                nn.Embedding(13073, 64),
                nn.Embedding(13170, 64),
                nn.Embedding(3145, 64),
                nn.Embedding(33843, 64),
                nn.Embedding(14304, 64),
                nn.Embedding(11, 64),
                nn.Embedding(13601, 64)
            ])

            self.numeric_embeddings = nn.ModuleList([
                nn.Embedding(64, 64),
                nn.Embedding(16, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(512, 64),
                nn.Embedding(512, 64)
            ])
            presize = 1088

        if name == "MLP_FSIW":
            print("using elapse feature")
            presize += 1

        self.mlp = nn.ModuleList([
            nn.Linear(presize, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        ])

        if self.model_name in ["MLP_SIG", "MLP_FSIW"]:
            self.mlp.append(nn.Linear(128, 1))
        else:
            raise ValueError("model name {} not exist".format(name))

    def forward(self, x):
        if self.dataset_source == "criteo":
            cate_embeddings = []
            nume_embeddings = []
            if self.model_name == "MLP_FSIW":
                for i in range(8):
                    nume_embeddings.append(self.numeric_embeddings[i].weight[x[:, i].long()])

                for i in range(9):
                    cate_embeddings.append(self.category_embeddings[8 - i].weight[x[:, -i - 2].long()])
                    
                features = nume_embeddings + cate_embeddings + [x[:,-1:]]
                x = torch.cat(features, dim = 1)
            else:
                for i in range(8):
                    nume_embeddings.append(self.numeric_embeddings[i].weight[x[:, i].long()])
                    
                for i in range(9):
                    cate_embeddings.append(self.category_embeddings[8 - i].weight[x[:, -i - 2].long()])

                features = nume_embeddings + cate_embeddings
                x = torch.cat(features, dim = 1)

        for layer in self.mlp:
            x = layer(x)
            
        if self.model_name in ["MLP_SIG", "MLP_FSIW"]:
            return {"logits": x}
        else:
            raise NotImplementedError()  
        
class DeepFM(nn.Module):
    def __init__(self, name, params):
        super(DeepFM, self).__init__()
        self.model_name = name
        self.params = params
        self.dataset_source = params["dataset_source"]
        if params["dataset_source"] == "criteo":
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(55824, 64),
                nn.Embedding(5443, 64),
                nn.Embedding(13073, 64),
                nn.Embedding(13170, 64),
                nn.Embedding(3145, 64),
                nn.Embedding(33843, 64),
                nn.Embedding(14304, 64),
                nn.Embedding(11, 64),
                nn.Embedding(13601, 64)
            ])

            self.numeric_embeddings = nn.ModuleList([
                nn.Embedding(64, 64),
                nn.Embedding(16, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(512, 64),
                nn.Embedding(512, 64)
            ])
            
            self.category_bias = nn.ParameterList([
                nn.Parameter(torch.zeros([55824, 1])),
                nn.Parameter(torch.zeros([5443, 1])),
                nn.Parameter(torch.zeros([13073, 1])),
                nn.Parameter(torch.zeros([13170, 1])),
                nn.Parameter(torch.zeros([3145, 1])),
                nn.Parameter(torch.zeros([33843, 1])),
                nn.Parameter(torch.zeros([14304, 1])),
                nn.Parameter(torch.zeros([11, 1])),
                nn.Parameter(torch.zeros([13601, 1]))
            ])

            self.numeric_bias = nn.ParameterList([
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([16, 1])),
                nn.Parameter(torch.zeros([128, 1])),
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([128, 1])),
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([512, 1])),
                nn.Parameter(torch.zeros([512, 1]))
            ])
            presize = 1088

        self.mlp = nn.ModuleList([
            nn.Linear(presize, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        ])


        self.mlp.append(nn.Linear(128, 1))
        
        self.init()
        
    def init(self):
        for i in range(len(self.category_embeddings)):
            nn.init.normal_(self.category_embeddings[i].weight, 0 , 0.01)
            
        for i in range(len(self.numeric_embeddings)):
            nn.init.normal_(self.numeric_embeddings[i].weight, 0 , 0.01)

    def forward(self, x):
        if self.dataset_source == "criteo":
            cate_embeddings = []
            nume_embeddings = []
            cate_bias = []
            nume_bias = []
            
            for i in range(8):
                nume_embeddings.append(self.numeric_embeddings[i].weight[x[:, i].long()])
                nume_bias.append(self.numeric_bias[i][x[:, i].long()])
                
            for i in range(9):
                cate_embeddings.append(self.category_embeddings[8 - i].weight[x[:, -i - 2].long()])
                cate_bias.append(self.category_bias[8 - i][x[:, -i - 2].long()])

            features = nume_embeddings + cate_embeddings
            x = torch.cat(features, dim = 1)
            
            bias = torch.cat(cate_bias + nume_bias, dim=1).sum(dim=1).unsqueeze(-1)
            features = [feat.unsqueeze(-2) for feat in features]
            fm_vectors = torch.cat(features, dim=1)
            fm_score = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2)).sum(dim=-1).unsqueeze(-1)

        for layer in self.mlp:
            x = layer(x)
            
        x = x + bias + fm_score

        return {"logits": x}
        
class AutoInt(nn.Module):
    def __init__(self, name, params):
        super(AutoInt, self).__init__()
        self.model_name = name
        self.params = params
        self.dataset_source = params["dataset_source"]
        
        self.n_layers = 3
        self.embedding_size = 64
        self.attention_size=64
        self.num_heads = 2
        
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        
        if params["dataset_source"] == "criteo":
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(55824, 64),
                nn.Embedding(5443, 64),
                nn.Embedding(13073, 64),
                nn.Embedding(13170, 64),
                nn.Embedding(3145, 64),
                nn.Embedding(33843, 64),
                nn.Embedding(14304, 64),
                nn.Embedding(11, 64),
                nn.Embedding(13601, 64)
            ])

            self.numeric_embeddings = nn.ModuleList([
                nn.Embedding(64, 64),
                nn.Embedding(16, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(512, 64),
                nn.Embedding(512, 64)
            ])
            
            self.category_bias = nn.ParameterList([
                nn.Parameter(torch.zeros([55824, 1])),
                nn.Parameter(torch.zeros([5443, 1])),
                nn.Parameter(torch.zeros([13073, 1])),
                nn.Parameter(torch.zeros([13170, 1])),
                nn.Parameter(torch.zeros([3145, 1])),
                nn.Parameter(torch.zeros([33843, 1])),
                nn.Parameter(torch.zeros([14304, 1])),
                nn.Parameter(torch.zeros([11, 1])),
                nn.Parameter(torch.zeros([13601, 1]))
            ])

            self.numeric_bias = nn.ParameterList([
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([16, 1])),
                nn.Parameter(torch.zeros([128, 1])),
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([128, 1])),
                nn.Parameter(torch.zeros([64, 1])),
                nn.Parameter(torch.zeros([512, 1])),
                nn.Parameter(torch.zeros([512, 1]))
            ])
            presize = 1088
            
        self.embed_output_dim = presize
        self.atten_output_dim = presize

        self.mlp_layers = nn.ModuleList([
            nn.Linear(presize, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        ])
        
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.attention_size, self.num_heads
                )
                for _ in range(self.n_layers)
            ]
        )
        
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        self.deep_predict_layer = nn.Linear(128, 1)

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, 0, 0.01)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                constant_(module.bias.data, 0)
                
    def autoint_layer(self, infeature):
        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Interacting layer
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        x = infeature.view(batch_size, -1)
        for layer in self.mlp_layers:
            x = layer(x)
        att_output = self.attn_fc(cross_term) + self.deep_predict_layer(x)
        return att_output

    def forward(self, x):
        if self.dataset_source == "criteo":
            cate_embeddings = []
            nume_embeddings = []
            cate_bias = []
            nume_bias = []
            
            for i in range(8):
                nume_embeddings.append(self.numeric_embeddings[i].weight[x[:, i].long()])
                nume_bias.append(self.numeric_bias[i][x[:, i].long()])
                
            for i in range(9):
                cate_embeddings.append(self.category_embeddings[8 - i].weight[x[:, -i - 2].long()])
                cate_bias.append(self.category_bias[8 - i][x[:, -i - 2].long()])

            features = nume_embeddings + cate_embeddings
            x = torch.cat(features, dim = 1)
            
            bias = torch.cat(cate_bias + nume_bias, dim=1).sum(dim=1).unsqueeze(-1)
            features = [feat.unsqueeze(-2) for feat in features]
            fm_vectors = torch.cat(features, dim=1)
            
        x = bias + self.autoint_layer(fm_vectors)

        return {"logits": x}
        
class DCNV2(nn.Module):
    def __init__(self, name, params):
        super(DCNV2, self).__init__()
        self.model_name = name
        self.params = params
        self.cross_layer_num = 1
        self.dataset_source = params["dataset_source"]
        if params["dataset_source"] == "criteo":
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(55824, 64),
                nn.Embedding(5443, 64),
                nn.Embedding(13073, 64),
                nn.Embedding(13170, 64),
                nn.Embedding(3145, 64),
                nn.Embedding(33843, 64),
                nn.Embedding(14304, 64),
                nn.Embedding(11, 64),
                nn.Embedding(13601, 64)
            ])

            self.numeric_embeddings = nn.ModuleList([
                nn.Embedding(64, 64),
                nn.Embedding(16, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(128, 64),
                nn.Embedding(64, 64),
                nn.Embedding(512, 64),
                nn.Embedding(512, 64)
            ])
            presize = 1088

        self.in_feature_num = presize
        
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num))
            for _ in range(self.cross_layer_num)
        )
        
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1))
            for _ in range(self.cross_layer_num)
        )

        self.mlp = nn.ModuleList([
            nn.Linear(presize, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        ])

        self.mlp.append(nn.Linear(128, 1))
    
        self.init()
        
    def init(self):
        for i in range(len(self.category_embeddings)):
            nn.init.normal_(self.category_embeddings[i].weight, 0 , 0.01)
            
        for i in range(len(self.numeric_embeddings)):
            nn.init.normal_(self.numeric_embeddings[i].weight, 0 , 0.01)
            
    def cross_network(self, x_0):
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l

        x_l = x_l.squeeze(dim=2)
        return x_l

    def forward(self, x):
        if self.dataset_source == "criteo":
            cate_embeddings = []
            nume_embeddings = []

            for i in range(8):
                nume_embeddings.append(self.numeric_embeddings[i].weight[x[:, i].long()])
                
            for i in range(9):
                cate_embeddings.append(self.category_embeddings[8 - i].weight[x[:, -i - 2].long()])

            features = nume_embeddings + cate_embeddings
            x = torch.cat(features, dim = 1)
            
            x = self.cross_network(x)

        for layer in self.mlp:
            x = layer(x)

        return {"logits": x}
        
class DFM(nn.Module):
    def __init__(self, name, params):
        super(DFM, self).__init__()
        self.model_name = name
        self.params = params
        
        if params["base_model"] == "MLP":
            self.CVR_MLP = MLP("MLP_SIG", params)
        elif params["base_model"] == "DeepFM":
            self.CVR_MLP = DeepFM("MLP_SIG", params)
        elif params["base_model"] == "DCNV2":
            self.CVR_MLP = DCNV2("MLP_SIG", params)
        elif params["base_model"] == "AutoInt":
            self.CVR_MLP = AutoInt("MLP_SIG", params)
        self.Delay_MLP = MLP("MLP_SIG", params)

    def forward(self, x):
        cvr_x = self.CVR_MLP(x)["logits"]
        delay_x = self.Delay_MLP(x)["logits"]
        
        return {"logits": torch.reshape(cvr_x, (-1, 1)), "log_lamb": torch.reshape(delay_x, (-1, 1))}

def get_model(name, params):
    if name in ["MLP_tn_dp", "MLP_FSIW"]:
        return MLP(name, params)
    elif name == "MLP_EXP_DELAY":
        return DFM(name, params)
    elif name == "MLP_SIG":
        if params["base_model"] == "MLP":
            return MLP(name, params)
        elif params["base_model"] == "DeepFM":
            return DeepFM(name, params)
        elif params["base_model"] == "DCNV2":
            return DCNV2(name, params)
        elif params["base_model"] == "AutoInt":
            return AutoInt(name, params)
    else:
        raise NotImplementedError()


