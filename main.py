import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics.pairwise import cosine_distances
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub
import timm

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Download all 3 datasets
# ----------------------------
busi_path = kagglehub.dataset_download("sabahesaraki/breast-ultrasound-images-dataset")
his_path  = kagglehub.dataset_download("orvile/hisbreast-breast-ultrasound-dataset")
uclm_path = kagglehub.dataset_download("orvile/bus-uclm-breast-ultrasound-dataset")

print(busi_path, his_path, uclm_path)

# ----------------------------
# Data collection
# ----------------------------
def collect_train_binary(busi_path, his_path):
    paths, labels = [], []
    lm = {"benign":0, "malignant":1}

    busi_root = os.path.join(busi_path, "Dataset_BUSI_with_GT")
    for cls in lm:
        d = os.path.join(busi_root, cls)
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            fl = f.lower()
            if fl.endswith(".png") and "mask" not in fl and "gt" not in fl:
                paths.append(os.path.join(d,f))
                labels.append(lm[cls])

    for root, dirs, _ in os.walk(his_path):
        for d in dirs:
            dl = d.lower()
            if dl in lm:
                dd = os.path.join(root, d)
                for f in os.listdir(dd):
                    if f.lower().endswith(".png"):
                        paths.append(os.path.join(dd,f))
                        labels.append(lm[dl])

    return paths, np.array(labels)

def collect_uclm_binary(uclm_path):
    paths, labels = [], []
    lm = {"benign":0, "malignant":1}
    root = os.path.join(uclm_path, "bus_uclm_separated")

    for cls in lm:
        d = os.path.join(root, cls)
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(".png"):
                paths.append(os.path.join(d,f))
                labels.append(lm[cls])

    return paths, np.array(labels)

train_paths, train_labels = collect_train_binary(busi_path, his_path)
uclm_paths, uclm_labels = collect_uclm_binary(uclm_path)

print("Train:", len(train_paths), "UCLM:", len(uclm_paths))

# ----------------------------
# Dataset 128x128
# ----------------------------
tfm = A.Compose([
    A.Resize(128,128),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

class USDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        img = img[...,None]
        img = tfm(image=img)["image"]
        return img, self.labels[i]

train_loader = DataLoader(
    USDataset(train_paths, train_labels),
    batch_size=16, shuffle=True
)

# ----------------------------
# VAE
# ----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,256,4,2,1), nn.ReLU()
        )
        self.fc_mu  = nn.Linear(256*8*8, latent_dim)
        self.fc_log = nn.Linear(256*8*8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim,256*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1)
        )

    def encode(self,x):
        h = self.enc(x).view(x.size(0),-1)
        return self.fc_mu(h), self.fc_log(h)

    def forward(self,x):
        mu, log = self.encode(x)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*log)
        out = self.dec(self.fc_dec(z).view(-1,256,8,8))
        return out, mu, log

vae = VAE(64).to(device)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

for ep in range(20):
    tot = 0
    for x,_ in train_loader:
        x = x.to(device)
        recon, mu, log = vae(x)
        loss = F.mse_loss(recon, x) + 0.1*torch.mean(mu**2)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"VAE Epoch {ep+1}: {tot/len(train_loader):.4f}")

vae.eval()

# ----------------------------
# Diffusion
# ----------------------------
class ImageDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,1,3,1,1)
        )
    def forward(self,x):
        return self.net(x)

diffusion = ImageDiffusion().to(device)
opt = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

for ep in range(15):
    tot = 0
    for x,_ in train_loader:
        x = x.to(device)
        loss = F.mse_loss(diffusion(x + 0.1*torch.randn_like(x)), x)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Diff Epoch {ep+1}: {tot/len(train_loader):.4f}")

# ----------------------------
# Fusion
# ----------------------------
class DiffFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        return self.net(x).view(x.size(0),-1)

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64+32,256), nn.ReLU(),
            nn.Linear(256,128)
        )
    def forward(self,a,b):
        return self.fc(torch.cat([a,b],1))

vae_feat  = lambda x: vae.encode(x)[0]
diff_feat = DiffFeat().to(device)
fusion    = FusionNet().to(device)

opt = torch.optim.Adam(fusion.parameters(), lr=1e-4)

for ep in range(20):
    tot = 0
    for x,_ in train_loader:
        x = x.to(device)
        emb = fusion(vae_feat(x), diff_feat(x))
        loss = torch.cdist(emb, emb).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Emb Epoch {ep+1}: {tot/len(train_loader):.4f}")

tfm = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

class USDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        img = img[...,None]
        img = tfm(image=img)["image"]
        return img, self.labels[i]

counts = np.bincount(train_labels)
weights = 1. / counts
sample_w = weights[train_labels]

sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

train_loader = DataLoader(
    USDataset(train_paths, train_labels),
    batch_size=32, sampler=sampler
)

# ----------------------------
# Backbone + Metric Head
# ----------------------------
backbone = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    in_chans=1,
    num_classes=0
).to(device)

for p in backbone.parameters():
    p.requires_grad = False

BACKBONE_DIM = backbone.num_features

class MetricHead(nn.Module):
    def __init__(self, in_dim=768, emb_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
    def forward(self,x):
        return F.normalize(self.fc(x), dim=1)

metric = MetricHead(BACKBONE_DIM, 256).to(device)

class BatchHardTriplet(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    def forward(self, emb, y):
        dist = torch.cdist(emb, emb)
        pos = y.unsqueeze(0) == y.unsqueeze(1)
        hardest_pos = (dist * pos.float()).max(dim=1)[0]
        hardest_neg = (dist + 1e6 * pos.float()).min(dim=1)[0]
        return F.relu(hardest_pos - hardest_neg + self.margin).mean()

criterion = BatchHardTriplet(0.3)
opt = torch.optim.Adam(metric.parameters(), lr=1e-4)

for ep in range(50):
    metric.train()
    tot = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            feat = backbone(x)
        emb = metric(feat)
        loss = criterion(emb,y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Epoch {ep+1:02d} | Triplet Loss: {tot/len(train_loader):.4f}")

# ----------------------------
# Retrieval
# ----------------------------
def get_emb(loader):
    E,L = [],[]
    metric.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            feat = backbone(x)
            emb = metric(feat)
            E.append(emb.cpu().numpy())
            L.append(y.numpy())
    return np.vstack(E), np.hstack(L)

def precision_at_5(Eq,Lq,Eg,Lg):
    d = cosine_distances(Eq,Eg)
    c=0
    for i in range(len(Eq)):
        idx = np.argsort(d[i])[:5]
        c += np.sum(Lg[idx]==Lq[i])
    return c/(len(Eq)*5)

# Cross-domain split
np.random.seed(42)
idx = np.arange(len(uclm_paths))
np.random.shuffle(idx)
mid = len(idx)//2

q_paths = [uclm_paths[i] for i in idx[:mid]]
g_paths = [uclm_paths[i] for i in idx[mid:]]

q_labels = uclm_labels[idx[:mid]]
g_labels = uclm_labels[idx[mid:]]

q_loader = DataLoader(USDataset(q_paths,q_labels), batch_size=32)
g_loader = DataLoader(USDataset(g_paths,g_labels), batch_size=32)

Eq,Lq = get_emb(q_loader)
Eg,Lg = get_emb(g_loader)

print("Cross-domain Precision@5:", precision_at_5(Eq,Lq,Eg,Lg))

# In-domain split
np.random.seed(42)
idx = np.arange(len(train_paths))
np.random.shuffle(idx)
mid = len(idx)//2

q_tr_paths = [train_paths[i] for i in idx[:mid]]
g_tr_paths = [train_paths[i] for i in idx[mid:]]

q_tr_labels = train_labels[idx[:mid]]
g_tr_labels = train_labels[idx[mid:]]

q_tr_loader = DataLoader(USDataset(q_tr_paths,q_tr_labels), batch_size=32)
g_tr_loader = DataLoader(USDataset(g_tr_paths,g_tr_labels), batch_size=32)

Eq_in,Lq_in = get_emb(q_tr_loader)
Eg_in,Lg_in = get_emb(g_tr_loader)

print("In-domain Precision@5:", precision_at_5(Eq_in,Lq_in,Eg_in,Lg_in))
