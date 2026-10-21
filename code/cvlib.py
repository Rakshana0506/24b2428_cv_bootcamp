"""Shared code for the CIFAR-10N notebooks: data loading, GPU augmentation, the training loop,
the robust losses and co-teaching, prediction files and results.json helpers.

Every run uses the same setup (SETUP below); only the method under test changes.
Images stay on the GPU as uint8, and random crop + flip are done on the GPU, so an epoch of
ResNet-18 on 45k images takes a few seconds on an A100.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- fixed training setup
EPOCHS = 60
BATCH = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 42
NUM_CLASSES = 10

SETUP = {
    "optimizer": "SGD (momentum 0.9, nesterov, weight decay 5e-4)",
    "learning_rate_schedule": "cosine annealing per step, 0.1 to 0 over 60 epochs, no warmup",
    "batch_size": BATCH,
    "epochs": EPOCHS,
    "augmentation": "random crop 32x32 with 4-pixel zero padding, random horizontal flip",
}

# CIFAR-10 channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# small-scale mode used only to smoke-test the notebooks on a CPU (CV_FAST=1)
FAST = os.environ.get("CV_FAST") == "1"
if FAST:
    EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- project folder
def find_root(start=None):
    """Folder that holds data/labels.csv and model.py."""
    cands = [Path(start)] if start else []
    cands += [Path.cwd(), Path.cwd().parent]
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        cands += [p.parent.parent for p in drive.glob("*/data/labels.csv")]
        cands += [p.parent.parent for p in drive.glob("*/*/data/labels.csv")]
    for p in cands:
        if (p / "data" / "labels.csv").exists() and (p / "model.py").exists():
            return p.resolve()
    return None


# ---------------------------------------------------------------- data
class Data:
    """All images as uint8 NCHW tensors on the GPU, plus the label table."""

    def __init__(self, root):
        root = Path(root)
        d = root / "data"
        self.root = root
        self.labels = pd.read_csv(d / "labels.csv").sort_values("index").reset_index(drop=True)
        assert (self.labels["index"].to_numpy() == np.arange(len(self.labels))).all()
        self.train_idx = np.loadtxt(d / "train_idx.txt", dtype=np.int64)
        self.val_idx = np.loadtxt(d / "val_idx.txt", dtype=np.int64)
        self.classes = [c.strip() for c in (d / "classes.txt").read_text().splitlines() if c.strip()]
        if FAST:      # tiny subset for smoke tests
            self.train_idx = self.train_idx[:1200]
            self.val_idx = self.val_idx[:400]
        self.x = _to_gpu(np.load(d / "train_images.npy"))
        self.x_test = _to_gpu(np.load(d / "test_images.npy"))
        if FAST:
            self.x_test = self.x_test[:10000]

    def y(self, column, idx):
        """Labels of one column ('noisy_label', 'synthetic_label', 'clean_label') for the given indices."""
        return self.labels[column].to_numpy()[idx]


def _to_gpu(a):
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(DEVICE)


_MEAN = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)


def normalize(xb):
    """uint8 or [0,1] float batch -> normalized float, channels_last."""
    if xb.dtype == torch.uint8:
        xb = xb.float().div_(255)
    return ((xb - _MEAN) / _STD).contiguous(memory_format=torch.channels_last)


def augment(xb, gen):
    """Random crop (4-pixel zero padding) and horizontal flip, per image, on the GPU."""
    n = xb.shape[0]
    x = F.pad(xb.float().div_(255), (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=xb.device, generator=gen)
    j = torch.randint(0, 9, (n,), device=xb.device, generator=gen)
    ar = torch.arange(32, device=xb.device)
    rows = (i.view(n, 1) + ar).view(n, 1, 32, 1)
    cols = (j.view(n, 1) + ar).view(n, 1, 1, 32)
    x = x[torch.arange(n, device=xb.device).view(n, 1, 1, 1), torch.arange(3, device=xb.device).view(1, 3, 1, 1), rows, cols]
    flip = torch.rand(n, device=xb.device, generator=gen) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    return normalize(x)


# ---------------------------------------------------------------- model
class _Tiny(nn.Module):
    """Stand-in network for CPU smoke tests only (CV_FAST=1)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.linear(self.f(x))


def make_model(root):
    if FAST:
        return _Tiny().to(DEVICE).to(memory_format=torch.channels_last)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model import ResNet18
    return ResNet18(num_classes=NUM_CLASSES).to(DEVICE).to(memory_format=torch.channels_last)


def _autocast():
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(net, x, bs=1000):
    """Softmax probabilities (float32 numpy) for a uint8 image tensor, no augmentation."""
    was_training = net.training
    net.eval()
    out = []
    for k in range(0, len(x), bs):
        with _autocast():
            out.append(F.softmax(net(normalize(x[k:k + bs])).float(), 1))
    net.train(was_training)
    return torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------- losses
def ce_loss(logits, y, **_):
    return F.cross_entropy(logits.float(), y, reduction="none")


def sce_loss(logits, y, alpha=0.1, beta=1.0, **_):
    """Symmetric cross entropy (Wang et al., ICCV 2019), CIFAR-10 setting alpha=0.1, beta=1.0.
    The reverse term swaps prediction and label, with log(0) clipped to log(1e-4)."""
    logits = logits.float()
    ce = F.cross_entropy(logits, y, reduction="none")
    p = F.softmax(logits, 1).clamp(min=1e-7, max=1.0)
    onehot = F.one_hot(y, logits.shape[1]).float().clamp(min=1e-4, max=1.0)
    rce = -(p * onehot.log()).sum(1)
    return alpha * ce + beta * rce


# ---------------------------------------------------------------- training
def _optimizer(net, steps_total):
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps_total) / steps_total)))
    return opt, sched


def train(data, idx, targets, *, seed=SEED, loss_fn=ce_loss, epochs=None, on_epoch=None, log_every=5, tag=""):
    """Train one ResNet-18 from scratch on images `idx` with integer labels `targets`.

    on_epoch(epoch, net) is called after every epoch (epoch counts from 1) and may return a dict
    that is printed every `log_every` epochs. Returns the trained network.
    """
    epochs = epochs or EPOCHS
    seed_everything(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    net = make_model(data.root)
    idx_t = torch.as_tensor(np.asarray(idx), device=DEVICE)
    y_t = torch.as_tensor(np.asarray(targets), device=DEVICE, dtype=torch.long)
    n = len(idx_t)
    steps = math.ceil(n / BATCH)
    opt, sched = _optimizer(net, steps * epochs)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=DEVICE, generator=gen)
        tot = torch.zeros((), device=DEVICE)
        for k in range(0, n, BATCH):
            b = perm[k:k + BATCH]
            xb = augment(data.x[idx_t[b]], gen)
            with _autocast():
                logits = net(xb)
            loss = loss_fn(logits, y_t[b], epoch=ep).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach() * len(b)
        info = on_epoch(ep, net) if on_epoch else None
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            extra = "  ".join(f"{k} {v:.4f}" for k, v in (info or {}).items())
            print(f"{tag} epoch {ep:2d}  loss {tot.item() / n:.4f}  {extra}  ({time.time() - t0:.0f}s)")
    return net


def train_coteaching(data, idx, targets, *, forget_rate, ramp_epochs=10, seed=SEED, epochs=None,
                     on_epoch=None, log_every=5, tag=""):
    """Co-teaching (Han et al., NeurIPS 2018): two networks see the same batches; each keeps the
    (1 - r(t)) fraction of samples with the smallest loss under its own predictions and passes those
    to the other network for the update. r(t) grows linearly from 0 to forget_rate over ramp_epochs.

    on_epoch(epoch, net1, net2) is called after every epoch. Returns (net1, net2).
    """
    epochs = epochs or EPOCHS
    seed_everything(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    net1 = make_model(data.root)
    seed_everything(seed + 1)
    net2 = make_model(data.root)
    idx_t = torch.as_tensor(np.asarray(idx), device=DEVICE)
    y_t = torch.as_tensor(np.asarray(targets), device=DEVICE, dtype=torch.long)
    n = len(idx_t)
    steps = math.ceil(n / BATCH)
    opt1, sch1 = _optimizer(net1, steps * epochs)
    opt2, sch2 = _optimizer(net2, steps * epochs)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        rate = forget_rate * min(1.0, (ep - 1) / ramp_epochs)
        net1.train(), net2.train()
        perm = torch.randperm(n, device=DEVICE, generator=gen)
        for k in range(0, n, BATCH):
            b = perm[k:k + BATCH]
            xb = augment(data.x[idx_t[b]], gen)
            yb = y_t[b]
            with _autocast():
                l1, l2 = net1(xb), net2(xb)
            loss1 = F.cross_entropy(l1.float(), yb, reduction="none")
            loss2 = F.cross_entropy(l2.float(), yb, reduction="none")
            keep = max(1, int(round((1 - rate) * len(b))))
            sel1 = torch.argsort(loss1.detach())[:keep]       # small-loss picks of net1 -> teach net2
            sel2 = torch.argsort(loss2.detach())[:keep]       # small-loss picks of net2 -> teach net1
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            (loss1[sel2].mean() + loss2[sel1].mean()).backward()
            opt1.step(), opt2.step(), sch1.step(), sch2.step()
        info = on_epoch(ep, net1, net2) if on_epoch else None
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            extra = "  ".join(f"{k} {v:.4f}" for k, v in (info or {}).items())
            print(f"{tag} epoch {ep:2d}  forget {rate:.2f}  {extra}  ({time.time() - t0:.0f}s)")
    return net1, net2


# ---------------------------------------------------------------- per-epoch recorder
class Recorder:
    """Keeps val and test probabilities for every epoch, so any epoch can be chosen afterwards
    (final epoch, or the early-stopping epoch picked on noisy validation accuracy)."""

    def __init__(self, data, val_labels):
        self.data = data
        self.val_labels = np.asarray(val_labels)
        self.val_probs, self.test_probs, self.val_acc = [], [], []

    def __call__(self, ep, *nets):
        pv = np.mean([predict(n, self.data.x[self.data.val_idx]) for n in nets], 0)
        pt = np.mean([predict(n, self.data.x_test) for n in nets], 0)
        self.val_probs.append(pv.astype(np.float16))
        self.test_probs.append(pt.astype(np.float16))
        acc = float((pv.argmax(1) == self.val_labels).mean())
        self.val_acc.append(acc)
        return {"val_acc": acc}

    def best_epoch(self):
        """1-based epoch with the highest validation accuracy (earliest on ties)."""
        return int(np.argmax(self.val_acc)) + 1

    def save(self, path, **extra):
        np.savez_compressed(path, val_probs=np.stack(self.val_probs), test_probs=np.stack(self.test_probs),
                            val_acc=np.array(self.val_acc), **extra)


# ---------------------------------------------------------------- outputs
def write_preds(path, probs):
    """id,pred file for the 10,000 test images."""
    pred = np.asarray(probs).argmax(1)
    assert len(pred) == 10000, len(pred)
    pd.DataFrame({"id": np.arange(len(pred)), "pred": pred}).to_csv(path, index=False)
    return pred


def load_results(work):
    p = Path(work) / "results_partial.json"
    return json.loads(p.read_text()) if p.exists() else {}


def update_results(work, **sections):
    """Merge values into work/results_partial.json (one level deep per task)."""
    res = load_results(work)
    for k, v in sections.items():
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k].update(v)
        else:
            res[k] = v
    (Path(work) / "results_partial.json").write_text(json.dumps(res, indent=2))
    return res
