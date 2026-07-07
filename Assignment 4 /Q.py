"""
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
Xu et al., 2016 — Soft Attention Implementation in PyTorch

Structure:
  - Encoder       : Pretrained ResNet-50, outputs (B, 14, 14, 2048) feature maps
  - Attention     : MLP + softmax over 196 spatial locations
  - Decoder       : LSTM that attends over image regions at each step
  - Training loop : Teacher forcing + CrossEntropyLoss
  - Inference     : Greedy decoding
  - Visualization : Attention map overlay (optional, requires matplotlib)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, random, json
from collections import Counter


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1.  VOCABULARY
class Vocabulary:
    """Maps words <-> indices.  Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3."""

    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, freq_threshold: int = 1):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, captions: list[str]):
        counter = Counter()
        for cap in captions:
            counter.update(cap.lower().split())
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption: str) -> list[int]:
        tokens = caption.lower().split()
        return (
            [self.START]
            + [self.word2idx.get(t, self.UNK) for t in tokens]
            + [self.END]
        )

    def decode(self, indices: list[int]) -> str:
        words = []
        for i in indices:
            w = self.idx2word.get(i, "<unk>")
            if w in ("<start>", "<pad>"):
                continue
            if w == "<end>":
                break
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


# 2.  DATASET  (dummy OR real COCO subset)
# ---------- 2a. Dummy dataset (always works, no downloads) ----------
class DummyDataset(Dataset):
    """
    Generates random 224×224 RGB images paired with simple fake captions.
    Useful for verifying the model runs end-to-end without downloading data.
    """

    CAPTIONS = [
        "a dog sitting on a bench",
        "a cat playing with a ball",
        "a bird flying over the ocean",
        "a person riding a bicycle",
        "a red car on the street",
        "children playing in the park",
        "a woman reading a book",
        "a man eating a sandwich",
        "a horse running in a field",
        "a group of people at a table",
    ]

    def __init__(self, vocab: Vocabulary, size: int = 200, img_size: int = 224):
        self.vocab = vocab
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random noise image (3×224×224)
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img = self.transform(img)

        cap_str = self.CAPTIONS[idx % len(self.CAPTIONS)]
        cap_ids = torch.tensor(self.vocab.encode(cap_str), dtype=torch.long)
        return img, cap_ids


def collate_fn(batch):
    """Pad captions in a batch to the same length."""
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)                        # (B, 3, H, W)
    lengths = [len(c) for c in caps]
    max_len = max(lengths)
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, (cap, l) in enumerate(zip(caps, lengths)):
        padded[i, :l] = cap
    return imgs, padded, torch.tensor(lengths)


# ---------- 2b. Real COCO subset helper (optional) ----------
def build_coco_subset(
    coco_img_dir: str,
    coco_ann_file: str,     # path to captions_train2017.json
    vocab: Vocabulary,
    max_samples: int = 5000,
    img_size: int = 224,
) -> Dataset:
    """
    Returns a Dataset backed by a small COCO slice.

    Usage:
        vocab = Vocabulary()
        # build vocab from all captions first, then call this
        ds = build_coco_subset("coco/train2017",
                               "coco/annotations/captions_train2017.json",
                               vocab, max_samples=2000)
    """
    with open(coco_ann_file) as f:
        data = json.load(f)

    # group captions by image id
    img_caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        img_caps.setdefault(ann["image_id"], []).append(ann["caption"])

    img_info = {img["id"]: img["file_name"] for img in data["images"]}
    samples = []
    for img_id, fname in list(img_info.items())[:max_samples]:
        if img_id not in img_caps:
            continue
        cap = img_caps[img_id][0]          # use first caption only
        path = os.path.join(coco_img_dir, fname)
        if os.path.exists(path):
            samples.append((path, cap))

    class _COCODataset(Dataset):
        def __init__(self, samples, vocab, img_size):
            self.samples = samples
            self.vocab = vocab
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, cap_str = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img = self.tf(img)
            cap_ids = torch.tensor(self.vocab.encode(cap_str), dtype=torch.long)
            return img, cap_ids

    return _COCODataset(samples, vocab, img_size)


# 3.  ENCODER  (ResNet-50, spatial features)
class Encoder(nn.Module):
    """
    Pretrained ResNet-50 with the final avgpool + fc layers removed.
    Input  : (B, 3, 224, 224)
    Output : (B, num_pixels, encoder_dim)  — num_pixels = 14*14 = 196
    """

    def __init__(self, encoded_img_size: int = 14, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Keep everything up to (and including) layer4; drop avgpool + fc
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pool so any input size → (B, 2048, enc_size, enc_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size, encoded_img_size))

        self.fine_tune(fine_tune)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, 3, 224, 224)
        feats = self.resnet(imgs)                   # (B, 2048, 7, 7) for 224px input
        feats = self.adaptive_pool(feats)           # (B, 2048, 14, 14)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1)          # (B, 14, 14, 2048)
        feats = feats.view(B, H * W, C)            # (B, 196, 2048)
        return feats

    def fine_tune(self, enable: bool):
        """Freeze/unfreeze ResNet weights (fine-tune only layer4 when enabled)."""
        for p in self.resnet.parameters():
            p.requires_grad = False
        if enable:
            for child in list(self.resnet.children())[-3:]:
                for p in child.parameters():
                    p.requires_grad = True


# 4.  SOFT ATTENTION
class SoftAttention(nn.Module):
    """
    Bahdanau-style additive attention (Xu et al., Eq. 4).

    e_ti = f_att(a_i, h_{t-1})          ← MLP scoring function
    α_ti = softmax(e_ti)                 ← attention weights  (B, num_pixels)
    ẑ_t  = Σ_i α_ti * a_i               ← context vector     (B, encoder_dim)

    Both image features and the hidden state are projected into a common
    attention dimension, then summed + tanh + linear → scalar score per region.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        # Project image features to attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Project decoder hidden state to attention space
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # Collapse attention space → scalar score per region
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Args:
            encoder_out    : (B, num_pixels, encoder_dim)
            decoder_hidden : (B, decoder_dim)
        Returns:
            context        : (B, encoder_dim)   weighted sum of features
            alpha          : (B, num_pixels)    attention weights
        """
        # (B, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # (B, 1, attention_dim)  — broadcast over spatial locations
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)

        # Combined score → (B, num_pixels)
        energy = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha = F.softmax(energy, dim=1)            # (B, num_pixels)

        # Weighted sum of encoder features
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


# 5.  DECODER  (LSTM with soft attention)
class DecoderWithAttention(nn.Module):
    """
    At each timestep t the LSTM input is:
        [word_embedding(y_{t-1}) ; context_vector_t]
    concatenated, giving a vector of size (embed_dim + encoder_dim).

    Hidden / cell state are initialised from the mean image feature
    via small learned linear layers (as in the paper).
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # LSTM input = [embedding ∥ context]
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Init hidden/cell from mean image feature
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Output projection: hidden → vocab logits
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        """Initialise h_0, c_0 from the mean of encoder features."""
        mean_enc = encoder_out.mean(dim=1)          # (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_enc))       # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_enc))       # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        """
        Teacher-forcing forward pass.

        Args:
            encoder_out : (B, num_pixels, encoder_dim)
            captions    : (B, max_len)  token ids  incl. <start> … <end>
            lengths     : (B,)          true caption lengths
        Returns:
            predictions : (B, max_len-1, vocab_size)  logits
            alphas      : (B, max_len-1, num_pixels)  attention maps
        """
        B = encoder_out.size(0)
        # Decode length = caption length - 1 (we predict next token each step,
        # so the last input is the token before <end>)
        decode_lengths = (lengths - 1).tolist()
        max_dec_len = max(decode_lengths)

        embeddings = self.dropout(self.embedding(captions))  # (B, max_len, embed_dim)

        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(B, max_dec_len, self.vocab_size).to(DEVICE)
        alphas = torch.zeros(B, max_dec_len, encoder_out.size(1)).to(DEVICE)

        for t in range(max_dec_len):
            # Process all samples (padding will be handled by masking loss)
            context, alpha = self.attention(encoder_out, h)

            # LSTM input: [word_embed ∥ context]
            lstm_in = torch.cat(
                [embeddings[:, t, :], context], dim=1
            )                                               # (B, embed+enc)

            h, c = self.lstm_cell(lstm_in, (h, c))

            preds = self.fc(self.dropout(h))  # (B, vocab)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas, decode_lengths


# 6.  TRAINING LOOP
def train_one_epoch(encoder, decoder, loader, enc_opt, dec_opt, criterion, grad_clip=5.0):
    encoder.train(); decoder.train()
    total_loss, total_tokens = 0.0, 0

    for imgs, captions, lengths in loader:
        imgs = imgs.to(DEVICE)
        captions = captions.to(DEVICE)
        lengths = lengths.to(DEVICE)

        # Sort by length (descending) — required for pack_padded / our loop
        lengths, sort_idx = lengths.sort(descending=True)
        imgs = imgs[sort_idx]
        captions = captions[sort_idx]

        # Forward
        enc_out = encoder(imgs)                              # (B, 196, 2048)
        predictions, alphas, decode_lengths = decoder(enc_out, captions, lengths)

        # Targets are captions shifted by 1 (we predict token[1..end])
        targets = captions[:, 1:]                           # (B, max_len-1)

        # Pack into 1-D tensors (ignore padding)
        # predictions: (B, max_dec, vocab)  targets: (B, max_dec)
        preds_packed = pack_padded(predictions, decode_lengths)
        tgts_packed = pack_padded(targets, decode_lengths)

        loss = criterion(preds_packed, tgts_packed)

        # Doubly stochastic attention regularisation (Eq. 14 in paper)
        # Encourages Σ_t α_ti ≈ 1 for each image region
        loss += 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        enc_opt.zero_grad(); dec_opt.zero_grad()
        loss.backward()
        # Clip gradients
        nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        enc_opt.step(); dec_opt.step()

        total_loss += loss.item() * sum(decode_lengths)
        total_tokens += sum(decode_lengths)

    return total_loss / total_tokens


def pack_padded(tensor: torch.Tensor, lengths: list) -> torch.Tensor:
    """Flatten a padded (B, T, *) tensor, keeping only valid positions."""
    packed = []
    for i, l in enumerate(lengths):
        packed.append(tensor[i, :l])
    return torch.cat(packed, dim=0)


# 7.  GREEDY INFERENCE
@torch.no_grad()
def caption_image_greedy(
    encoder, decoder, img_tensor: torch.Tensor,
    vocab: Vocabulary, max_len: int = 40
) -> tuple[str, list]:
    """
    Generate a caption for a single image using greedy decoding.

    Returns:
        caption (str)
        attention_maps (list of (num_pixels,) arrays) — one per generated token
    """
    encoder.eval(); decoder.eval()

    img = img_tensor.unsqueeze(0).to(DEVICE)        # (1, 3, H, W)
    enc_out = encoder(img)                           # (1, 196, 2048)

    h, c = decoder.init_hidden_state(enc_out)

    word_id = torch.tensor([vocab.START]).to(DEVICE) # (1,)
    caption_ids = []
    attention_maps = []

    for _ in range(max_len):
        emb = decoder.embedding(word_id)             # (1, embed_dim)
        context, alpha = decoder.attention(enc_out, h)  # context:(1,enc_dim), alpha:(1,196)
        lstm_in = torch.cat([emb, context], dim=1)   # (1, embed+enc)
        h, c = decoder.lstm_cell(lstm_in, (h, c))
        logits = decoder.fc(h)                       # (1, vocab_size)
        word_id = logits.argmax(dim=1)               # greedy pick

        attention_maps.append(alpha.squeeze(0).cpu().numpy())
        token = word_id.item()
        if token == vocab.END:
            break
        caption_ids.append(token)

    return vocab.decode(caption_ids), attention_maps


# 8.  ATTENTION VISUALISATION  (optional)
def visualize_attention(
    img_tensor: torch.Tensor,
    caption_words: list[str],
    attention_maps: list,
    enc_size: int = 14,
    save_path: str = "attention_vis.png",
):
    """
    Overlay attention heat-maps on the image for each generated word.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from skimage.transform import resize
    except ImportError:
        print("Install matplotlib and scikit-image for visualisation.")
        return

    # Un-normalise image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)

    n = len(caption_words)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, (word, att) in enumerate(zip(caption_words, attention_maps)):
        att_map = att.reshape(enc_size, enc_size)
        att_up = resize(att_map, img.shape[:2], anti_aliasing=True)
        axes[i].imshow(img)
        axes[i].imshow(att_up, alpha=0.5, cmap=cm.hot)
        axes[i].set_title(word, fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Attention visualisation saved to {save_path}")
    plt.show()

def main():
    #Hyper-parameters 
    EMBED_DIM = 256
    ATTENTION_DIM = 256
    DECODER_DIM = 512
    ENCODER_DIM = 2048
    DROPOUT = 0.5
    BATCH_SIZE = 16
    EPOCHS = 5
    LR_ENC = 1e-4
    LR_DEC = 4e-4
    ENC_SIZE = 14           # spatial grid: 14×14 = 196 regions

    #Vocabulary 
    vocab = Vocabulary(freq_threshold=1)
    from dummy_captions import DummyDataset  # local ref — defined above
    vocab.build(DummyDataset.CAPTIONS)
    print(f"Vocabulary size: {len(vocab)}")

    #Dataset / Loader 
    # Swap DummyDataset with build_coco_subset(...) for real data
    train_ds = DummyDataset(vocab, size=256)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )

    #Model
    encoder = Encoder(encoded_img_size=ENC_SIZE, fine_tune=True).to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    #Optimizers 
    enc_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=LR_ENC
    )
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=LR_DEC)

    #Loss 
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD).to(DEVICE)

    #Training 
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(encoder, decoder, train_loader, enc_opt, dec_opt, criterion)
        print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {loss:.4f}")

    #Inference (greedy decoding) 
    print("\n── Inference ──")
    sample_img, _ = train_ds[0]          # grab a training image for demo
    caption, att_maps = caption_image_greedy(encoder, decoder, sample_img, vocab)
    print(f"Generated caption: {caption}")

    #Attention visualization 
    caption_words = caption.split()
    visualize_attention(sample_img, caption_words, att_maps[:len(caption_words)], enc_size=ENC_SIZE)


if __name__ == "__main__":
    # Quick hack so the local class reference in main() resolves
    import sys, types
    m = types.ModuleType("dummy_captions")
    m.DummyDataset = DummyDataset
    sys.modules["dummy_captions"] = m
    main()
