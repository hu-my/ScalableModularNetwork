import torch
import torch.nn as nn
from model.template import Template

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, args['embed_dim'], kernel_size=args['patch_size'], stride=args['patch_size'])  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args['embed_dim']), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args['img_size'] // args['patch_size']) ** 2 + 1, args['embed_dim']), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], self.args['embed_dim'], -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E

        x = torch.cat((torch.repeat_interleave(self.cls_token, int(x.shape[0]), 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args['n_attention_heads']
        self.embed_dim = args['embed_dim']
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

    def forward(self, x):
        m, s, e = x.shape

        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
        xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
        xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE

        xq = xq.reshape([-1, s, self.head_embed_dim])  # B, H, Q, HE -> (BH), Q, HE
        xk = xk.reshape([-1, s, self.head_embed_dim])  # B, H, K, HE -> (BH), K, HE
        xv = xv.reshape([-1, s, self.head_embed_dim])  # B, H, V, HE -> (BH), V, HE

        xk = xk.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K
        x_attention = xq.bmm(xk)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(xv)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  # (BH), Q, HE -> B, H, Q, HE
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.fc1 = nn.Linear(args['embed_dim'], args['embed_dim'] * args['forward_mul'])
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(args['embed_dim'] * args['forward_mul'], args['embed_dim'])
        self.norm1 = nn.LayerNorm(args['embed_dim'])
        self.norm2 = nn.LayerNorm(args['embed_dim'])

    def forward(self, x):
        x_ = self.attention(x)
        x = x + x_ # Skip connection
        x = self.norm1(x) # Normalization

        x_ = self.fc1(x)
        x_ = self.activation(x_)
        x_ = self.fc2(x_)
        x = x + x_ # Skip connection
        x = self.norm2(x) # Normalization
        return x


class Classifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(args['embed_dim'], args['embed_dim'])
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args['embed_dim'], num_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class VisionTransformer(Template):
    def __init__(self, args):
        super(VisionTransformer, self).__init__(args)
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args['n_layers'])], nn.LayerNorm(args['embed_dim']))
        self.classifier = Classifier(args, args['num_classes'])

    def forward_feature(self, x):
        x = x.unsqueeze(1).float()
        x = self.embedding(x)
        features = self.encoder(x)
        return features

    def forward_classifier(self, features, y=None, classifier=None):
        # the argument of classifier is used for finetuning new tasks
        if classifier is None:
            classifier = self.classifier

        preds = classifier(features)
        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim=-1)(preds)
            entropy = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            loss = self.Loss(preds, y) - entropy
            return probs, loss
        return preds

    def forward(self, x, y=None):
        features = self.forward_feature(x)
        return self.forward_classifier(features, y)

    def finetune_classifier(self, num_classes=10):
        classifier = Classifier(self.args, num_classes).cuda()
        return classifier