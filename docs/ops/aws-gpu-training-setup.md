# AWS GPU training setup (EC2 g5.xlarge)

Plain-English guide: rent a GPU computer on AWS, copy your code + face photos there, train faster than on a Mac, copy results back.

**This is for training experiments only.** Production model serving (SageMaker endpoint) is a separate step later.

---

## What you're doing (big picture)

1. **Launch** a cloud computer with an NVIDIA GPU (`g5.xlarge`).
2. **Copy** your repo (code) and `data/exports/` (face photos + labels) onto its disk.
3. **SSH in** and run the same `python scripts/train.py` you run on your Mac.
4. **Copy back** `artifacts/` and `checkpoints/` when done.
5. **Stop** the instance so you stop paying (~$1/hr while it runs).

Nothing goes public on the internet. Data sits on a private disk in your AWS account.

---

## Part 1 — Launch the EC2 instance (AWS Console)

### 1.1 Open EC2

1. Log in to [AWS Console](https://console.aws.amazon.com).
2. Search **EC2** → **Launch instance**.

### 1.2 Name

- **Name:** `faceiq-train-gpu`

### 1.3 AMI (operating system + PyTorch + GPU drivers)

You already picked the right one:

- **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04)**
- Username when you SSH: **`ubuntu`**

This AMI already has CUDA and PyTorch — you skip the hardest install steps.

### 1.4 Instance type

- **g5.xlarge** (1× NVIDIA A10G GPU, 24 GB VRAM, ~$1/hr)

If AWS says "capacity unavailable", try another **Availability Zone** in the same region (edit subnet in Network settings).

### 1.5 Key pair (required — do not skip)

1. **Create new key pair**
2. Name: e.g. `faceiq-gpu`
3. Type: **RSA**, format: **`.pem`**
4. Click **Create** → browser downloads `faceiq-gpu.pem`
5. Move it somewhere safe, e.g. `~/Downloads/faceiq-gpu.pem`

You need this file to SSH in. If you lose it, you cannot connect.

### 1.6 Network / security group

- **Auto-assign public IP:** Enable
- **Security group:** Create new
- **SSH (port 22):** change **Source** from `Anywhere (0.0.0.0/0)` to **My IP** (your home/office IP only)

Leave HTTP/HTTPS unchecked — you're not running a web server.

### 1.7 Storage

- **Root volume:** increase to **80–100 GiB** gp3 (30 GiB can work but gets tight with PyTorch + venv + export + checkpoints)

The summary may also show a large **instance store** volume (~250 GiB) — that's bonus fast disk on g5; the root EBS volume is what matters for a simple setup.

### 1.8 Launch

Click **Launch instance**. Wait until **Instance state** = **Running**.

### 1.9 Copy the public IP

EC2 → **Instances** → click `faceiq-train-gpu` → copy **Public IPv4 address** (e.g. `54.123.45.67`).

You'll use this as `GPU_IP` below.

---

## Part 2 — Prepare your Mac (one-time per key file)

Open **Terminal** on your Mac (not inside the GPU yet).

### 2.1 Fix key permissions

macOS requires the `.pem` file to be private:

```bash
chmod 400 ~/Downloads/faceiq-gpu.pem
```

### 2.2 Set shell variables (edit the IP each time you start a stopped instance)

```bash
export GPU_IP=54.123.45.67          # paste your instance Public IPv4
export GPU_KEY=~/Downloads/faceiq-gpu.pem
export GPU_USER=ubuntu
export REPO=/Users/ditmarhoxha/Developer/GitHub/faceiq-preference-ml
```

**Note:** If you **stop and start** the instance, the public IP often **changes** — update `GPU_IP`.

### 2.3 Test SSH

```bash
ssh -i $GPU_KEY $GPU_USER@$GPU_IP
```

First time it asks to trust the host — type `yes`.

You should see an Ubuntu prompt. Type `exit` to return to your Mac.

If connection fails: check security group allows SSH from your IP, instance is Running, and IP is correct.

---

## Part 3 — Copy repo + export to the GPU disk (run on your Mac)

These commands **push files from your Mac to the EC2 disk** via `rsync` over SSH.

### 3.1 Copy code (exclude heavy/local-only folders)

```bash
rsync -avz --progress \
  -e "ssh -i $GPU_KEY" \
  --exclude '.venv' \
  --exclude 'checkpoints' \
  --exclude 'artifacts' \
  --exclude 'data' \
  --exclude '.git' \
  --exclude '__pycache__' \
  $REPO/ \
  $GPU_USER@$GPU_IP:~/faceiq-preference-ml/
```

**What this does:** copies Python source, configs, scripts — not old checkpoints or your Mac venv.

### 3.2 Copy training data (face photos + JSONL labels)

```bash
rsync -avz --progress \
  -e "ssh -i $GPU_KEY" \
  $REPO/data/exports/ \
  $GPU_USER@$GPU_IP:~/faceiq-preference-ml/data/exports/
```

**What this does:** copies `manifest.json`, `faces.jsonl`, `matchups/`, and `images/` (~3k webp files). This may take 5–20 minutes depending on upload speed.

**Where files live on AWS:** `/home/ubuntu/faceiq-preference-ml/` on that instance's EBS disk — private to your account.

---

## Part 4 — SSH in and set up training (run on the GPU box)

```bash
ssh -i $GPU_KEY $GPU_USER@$GPU_IP
```

All commands below run **on the remote machine** (prompt looks like `ubuntu@ip-172-...`).

### 4.1 Verify GPU

```bash
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Expect `CUDA: True` and an NVIDIA A10G in `nvidia-smi`.

### 4.2 Create venv and install dependencies

```bash
cd ~/faceiq-preference-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install insightface onnxruntime onnx2torch
```

### 4.3 Run training (same configs as Mac)

```bash
source .venv/bin/activate
cd ~/faceiq-preference-ml

# Example: ArcFace long run
python scripts/train.py --config configs/train-v8-arcface-e2e-long.yaml

# After training finishes, evaluate vs BT
python scripts/evaluate.py \
  --checkpoint checkpoints/train-v8-arcface-e2e-long/best.pt \
  --ratings artifacts/bt-refit-v1/ratings.csv
```

**BT ratings for eval:** if you didn't copy `artifacts/bt-refit-v1/`, either rsync it from Mac or re-run BT on the GPU box once:

```bash
python scripts/run_bt.py --export data/exports/cmr1mr0m7000196d57zi3vcgn
```

### 4.4 Keep training running if you disconnect (recommended)

```bash
tmux new -s train
source .venv/bin/activate
cd ~/faceiq-preference-ml
python scripts/train.py --config configs/train-v8-arcface-e2e-long.yaml 2>&1 | tee train.log
```

- Detach: **Ctrl+B**, then **D**
- Reattach later: `tmux attach -t train`

---

## Part 5 — Copy results back to your Mac (run on your Mac)

When training finishes, **on your Mac** (set `GPU_IP` again if needed):

```bash
export GPU_IP=54.123.45.67
export GPU_KEY=~/Downloads/faceiq-gpu.pem
export GPU_USER=ubuntu
export REPO=/Users/ditmarhoxha/Developer/GitHub/faceiq-preference-ml

# Metrics, eval JSON, logs
rsync -avz --progress \
  -e "ssh -i $GPU_KEY" \
  $GPU_USER@$GPU_IP:~/faceiq-preference-ml/artifacts/ \
  $REPO/artifacts/

# Model weights (best.pt)
rsync -avz --progress \
  -e "ssh -i $GPU_KEY" \
  $GPU_USER@$GPU_IP:~/faceiq-preference-ml/checkpoints/ \
  $REPO/checkpoints/
```

View runs locally:

```bash
cd $REPO
source .venv/bin/activate
streamlit run app/dashboard.py --server.port 8502
```

Open **Training runs** tab — GPU runs appear alongside Mac runs.

---

## Part 6 — Stop billing

EC2 → Instances → select `faceiq-train-gpu`:

- **Stop instance** — pauses compute (~$1/hr stops); disk kept; IP may change on restart.
- **Terminate** — deletes everything (use when done for good).

Always stop/terminate when not training.

---

## Optional — Use Cursor on the GPU box

1. Cursor → Command Palette → **Remote-SSH: Connect to Host**
2. Add to `~/.ssh/config`:

```
Host faceiq-gpu
  HostName 54.123.45.67
  User ubuntu
  IdentityFile ~/Downloads/faceiq-gpu.pem
```

3. Connect → **Open Folder** → `/home/ubuntu/faceiq-preference-ml`
4. Terminal in Cursor now runs on the GPU. Same commands as Part 4.

You still **rsync from Mac** for the initial data copy (Part 3) unless you clone from git and rsync only `data/exports/`.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| SSH timeout | Security group → SSH from **My IP**; instance **Running** |
| `Permission denied (publickey)` | `chmod 400` on `.pem`; correct `GPU_USER=ubuntu` |
| `CUDA: False` | Wrong AMI or wrong instance type (need g5, not t3) |
| Out of disk | Increase root volume to 100 GiB or `df -h` and clean `.venv` |
| ArcFace download fails | First train run downloads InsightFace weights; needs internet |
| IP changed after stop/start | Update `GPU_IP` in rsync/ssh commands |

---

## SageMaker vs this setup

| | **EC2 + rsync (this guide)** | **SageMaker Training** |
|---|---|---|
| Best for | Fast manual experiments | Automated repeat pipelines |
| Setup time | ~15 minutes | Hours (S3, IAM, job configs) |
| Run new config | SSH + one command | New training job each time |

Use **EC2 for training experiments**. Use **SageMaker endpoint** later when the app needs live inference.

---

## Quick reference — full first-time flow

```bash
# === ON MAC ===
chmod 400 ~/Downloads/faceiq-gpu.pem
export GPU_IP=<your-public-ip>
export GPU_KEY=~/Downloads/faceiq-gpu.pem
export GPU_USER=ubuntu
export REPO=/Users/ditmarhoxha/Developer/GitHub/faceiq-preference-ml

rsync -avz -e "ssh -i $GPU_KEY" --exclude '.venv' --exclude 'checkpoints' \
  --exclude 'artifacts' --exclude 'data' --exclude '.git' \
  $REPO/ $GPU_USER@$GPU_IP:~/faceiq-preference-ml/

rsync -avz -e "ssh -i $GPU_KEY" \
  $REPO/data/exports/ $GPU_USER@$GPU_IP:~/faceiq-preference-ml/data/exports/

ssh -i $GPU_KEY $GPU_USER@$GPU_IP

# === ON GPU BOX ===
cd ~/faceiq-preference-ml && python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install insightface onnxruntime onnx2torch
python scripts/train.py --config configs/train-v7-arcface-e2e.yaml
exit

# === BACK ON MAC ===
rsync -avz -e "ssh -i $GPU_KEY" $GPU_USER@$GPU_IP:~/faceiq-preference-ml/artifacts/ $REPO/artifacts/
rsync -avz -e "ssh -i $GPU_KEY" $GPU_USER@$GPU_IP:~/faceiq-preference-ml/checkpoints/ $REPO/checkpoints/
```

Expected speed: **~30–60 min** per ArcFace e2e run vs **~5–10 hours** on Mac MPS.
