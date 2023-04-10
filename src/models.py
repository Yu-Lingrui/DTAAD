import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import dgl.nn
from dgl.nn.pytorch import GATConv
from torch.nn import TransformerEncoder
from src.gltcn import *
from src.dlutils import *
from src.constants import *

torch.manual_seed(1)
torch.cuda.manual_seed(1)


## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
    def __init__(self, feats):
        super(LSTM_Univariate, self).__init__()
        self.name = 'LSTM_Univariate'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 1
        self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

    def forward(self, x):
        hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64),
                   torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
        outputs = []
        for i, g in enumerate(x):
            multivariate_output = []
            for j in range(self.n_feats):
                univariate_input = g.view(-1)[j].view(1, 1, -1)
                out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
                multivariate_output.append(2 * out.view(-1))
            output = torch.cat(multivariate_output)
            outputs.append(output)
        return torch.stack(outputs)


## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
    def __init__(self, feats):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5  # MHA w_size = 5
        self.n = self.n_feats * self.n_window
        self.atts = [nn.Sequential(nn.Linear(self.n, feats * feats),
                                   nn.ReLU(True)) for i in range(1)]
        self.atts = nn.ModuleList(self.atts)

    def forward(self, g):
        for at in self.atts:
            ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
            g = torch.matmul(g, ats)
        return g, ats


## LSTM_NDT Model
class LSTM_NDT(nn.Module):
    def __init__(self, feats):
        super(LSTM_NDT, self).__init__()
        self.name = 'LSTM_NDT'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden)
        self.lstm2 = nn.LSTM(feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (
            torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
        hidden2 = (
            torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
        outputs = []
        for i, g in enumerate(x):
            out, hidden = self.lstm(g.view(1, 1, -1), hidden)
            out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
            out = self.fcn(out.view(-1))
            outputs.append(2 * out.view(-1))
        return torch.stack(outputs)


## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 5  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden


## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5  # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1, -1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        self.g = dgl.graph((torch.tensor(list(range(1, feats + 1))), torch.tensor([0] * feats)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(feats, 1, feats)
        self.time_gat = GATConv(feats, 1, feats)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1)

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats), data))
        feat_r = self.feature_gat(self.g, data_r)
        data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
        time_r = self.time_gat(self.g, data_t)
        data = torch.cat((torch.zeros(1, self.n_feats), data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        x, h = self.gru(x, hidden)
        return x.view(-1), h


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5  # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)

    
# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window 
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window) 
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2  # 广播机制
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2  

    
# Proposed Model + Tcn_Global + Transformer + MAML
class DTAAD_Tcn_Local(nn.Module):
    def __init__(self, feats):
        super(DTAAD_Tcn_Local, self).__init__()
        self.name = 'DTAAD_Tcn_Local'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)  # only one layer
        self.fcn = nn.Linear(feats, feats)
        self.decoder = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def forward(self, src):
        g_atts = self.g_tcn(src)
        src = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        z = self.transformer_encoder(src)
        c = z + self.fcn(z)
        x = self.decoder(c.permute(1, 2, 0))
        return x.permute(0, 2, 1)  # (Batch, 1, output_channel)


# Proposed Model + Tcn_Local + Transformer + MAML
class DTAAD_Tcn_Global(nn.Module):
    def __init__(self, feats): 
        super(DTAAD_Tcn_Global, self).__init__()
        self.name = 'DTAAD_Tcn_Global'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        z = self.transformer_encoder(src)
        c = z + self.fcn(z)
        x = self.decoder(c.permute(1, 2, 0))
        return x.permute(0, 2, 1)


# Proposed Model + Tcn_Local + Tcn_Global + Transformer + MAML
class DTAAD_Callback(nn.Module):
    def __init__(self, feats):
        super(DTAAD_Callback, self).__init__()
        self.name = 'DTAAD_Callback'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        g_atts = self.g_tcn(src)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        z2 = self.transformer_encoder2(src2)
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)


# Proposed Model + Tcn_Local + Tcn_Global + Callback + MAML
class DTAAD_Transformer(nn.Module):
    def __init__(self, feats):
        super(DTAAD_Transformer, self).__init__()
        self.name = 'DTAAD_Transformer'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)

        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        memory = self.g_tcn(src2).permute(0, 2, 1)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src).permute(0, 2, 1)
        c1 = l_atts + self.fcn(l_atts)
        x1 = self.decoder1(c1.permute(0, 2, 1))
        c2 = self.callback(src, x1)
        x2 = self.decoder2((c2 + self.fcn(c2)).permute(0, 2, 1))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)


# Proposed Model + Tcn_Local + Tcn_Global + Callback + Transformer + MAML
class DTAAD(nn.Module):
    def __init__(self, feats):
        super(DTAAD, self).__init__()
        self.name = 'DTAAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)
