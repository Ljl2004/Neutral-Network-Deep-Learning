"""
Complete experiment script for Project 1.
Vectorized conv2D makes CNN training fast.
"""
import matplotlib
matplotlib.use('Agg')
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os, time

save_dir = './experiment_results'
os.makedirs(save_dir, exist_ok=True)

print("Loading MNIST...")
np.random.seed(309)

with gzip.open(r'.\dataset\MNIST\train-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(r'.\dataset\MNIST\train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)
with gzip.open(r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(train_imgs.shape[0]))
train_imgs = train_imgs[idx] / 255.0
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]
test_imgs = test_imgs / 255.0
train_set = [train_imgs, train_labs]
valid_set = [valid_imgs, valid_labs]
test_set = [test_imgs, test_labs]
print(f"Train: {train_imgs.shape}, Valid: {valid_imgs.shape}, Test: {test_imgs.shape}")

results = {}

# ============================================================
# Part A: MLP Baseline (5 epochs)
# ============================================================
print("\n" + "="*60)
print("Part A: MLP Baseline (5 epochs)")
print("="*60)

mlp = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
mlp_opt = nn.optimizer.SGD(init_lr=0.06, model=mlp)
mlp_sched = nn.lr_scheduler.MultiStepLR(optimizer=mlp_opt, milestones=[800, 2400, 4000], gamma=0.5)
mlp_loss = nn.op.MultiCrossEntropyLoss(model=mlp, max_classes=10)
mlp_runner = nn.runner.RunnerM(mlp, mlp_opt, nn.metric.accuracy, mlp_loss, batch_size=64, scheduler=mlp_sched)

t0 = time.time()
mlp_runner.train(train_set, valid_set, num_epochs=5, log_iters=300, save_dir=save_dir)
mlp_time = time.time() - t0

mlp2 = nn.models.Model_MLP()
mlp2.load_model(os.path.join(save_dir, 'best_model.pickle'))
mlp_test = nn.metric.accuracy(mlp2(test_set[0]), test_set[1])
results['MLP'] = {'best_dev': mlp_runner.best_score, 'test_acc': mlp_test, 'time': mlp_time}
print(f"MLP: Dev={mlp_runner.best_score:.4f}, Test={mlp_test:.4f}, Time={mlp_time:.1f}s")

# Save MLP learning curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot(mlp_runner, axes)
axes[0].set_title("MLP Learning Curves")
plt.savefig(os.path.join(save_dir, 'mlp_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

np.savez(os.path.join(save_dir, 'mlp_history.npz'),
         train_loss=mlp_runner.train_loss, train_scores=mlp_runner.train_scores,
         dev_loss=mlp_runner.dev_loss, dev_scores=mlp_runner.dev_scores,
         best_score=mlp_runner.best_score, test_acc=mlp_test, train_time=mlp_time)

# Save MLP model separately
mlp.save_model(os.path.join(save_dir, 'mlp_best.pickle'))

# ============================================================
# Part B: CNN (5 epochs)
# ============================================================
print("\n" + "="*60)
print("Part B: CNN (5 epochs)")
print("="*60)

cnn = nn.models.Model_CNN(
    conv_config=[{'out_channels': 4, 'kernel_size': 3}],
    linear_config=[10],
    input_shape=(1, 28, 28)
)
cnn_params = sum(l.W.size + l.b.size for l in cnn.layers if l.optimizable)
print(f"CNN params: {cnn_params:,}")
print(f"Architecture: Conv2D(1,4,k=3,ReLU) -> Flatten -> Linear(2704,10)")

cnn_opt = nn.optimizer.SGD(init_lr=0.06, model=cnn)
cnn_sched = nn.lr_scheduler.MultiStepLR(optimizer=cnn_opt, milestones=[800, 2400, 4000], gamma=0.5)
cnn_loss = nn.op.MultiCrossEntropyLoss(model=cnn, max_classes=10)
cnn_runner = nn.runner.RunnerM(cnn, cnn_opt, nn.metric.accuracy, cnn_loss, batch_size=64, scheduler=cnn_sched)

t0 = time.time()
cnn_runner.train(train_set, valid_set, num_epochs=5, log_iters=300, save_dir=save_dir)
cnn_time = time.time() - t0

cnn2 = nn.models.Model_CNN()
cnn2.load_model(os.path.join(save_dir, 'best_model.pickle'))
cnn_test = nn.metric.accuracy(cnn2(test_set[0]), test_set[1])
results['CNN'] = {'best_dev': cnn_runner.best_score, 'test_acc': cnn_test,
                  'time': cnn_time, 'params': cnn_params}
print(f"CNN: Dev={cnn_runner.best_score:.4f}, Test={cnn_test:.4f}, Time={cnn_time:.1f}s")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot(cnn_runner, axes)
axes[0].set_title("CNN Learning Curves")
plt.savefig(os.path.join(save_dir, 'cnn_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

np.savez(os.path.join(save_dir, 'cnn_history.npz'),
         train_loss=cnn_runner.train_loss, train_scores=cnn_runner.train_scores,
         dev_loss=cnn_runner.dev_loss, dev_scores=cnn_runner.dev_scores,
         best_score=cnn_runner.best_score, test_acc=cnn_test, train_time=cnn_time, n_params=cnn_params)

# ============================================================
# MLP vs CNN comparison plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
ax = axes[0]
for name, data, color in [('MLP', mlp_runner, '#E3E37D'), ('CNN', cnn_runner, '#968A62')]:
    dev_x = np.linspace(0, len(data.train_loss)-1, len(data.dev_loss))
    ax.plot(dev_x, data.dev_loss, label=f'{name} (Dev)', color=color, linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('MLP vs CNN: Dev Loss')
ax.legend()

# Accuracy comparison
ax = axes[1]
for name, data, color in [('MLP', mlp_runner, '#E3E37D'), ('CNN', cnn_runner, '#968A62')]:
    dev_x = np.linspace(0, len(data.train_scores)-1, len(data.dev_scores))
    ax.plot(dev_x, data.dev_scores, label=f'{name} (Dev)', color=color, linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.set_title('MLP vs CNN: Dev Accuracy')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mlp_vs_cnn.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Part C Direction 1: Optimization Comparison (3 epochs each)
# ============================================================
print("\n" + "="*60)
print("Part C Direction 1: Optimization Comparison (3 epochs)")
print("="*60)

opt_configs = [
    ('SGD (baseline)', 'SGD', 0.06, None, 'multistep'),
    ('SGD (no scheduler)', 'SGD', 0.06, None, None),
    ('Momentum (mu=0.9)', 'MomentGD', 0.06, 0.9, 'multistep'),
    ('Momentum + ExponentialLR', 'MomentGD', 0.06, 0.9, 'exponential'),
]

opt_results = {}
for name, opt_name, lr, mu, sched_type in opt_configs:
    print(f"\n--- {name} ---")
    np.random.seed(309)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])

    if opt_name == 'SGD':
        optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    else:
        optimizer = nn.optimizer.MomentGD(init_lr=lr, model=model, mu=mu)

    scheduler = None
    if sched_type == 'multistep':
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500, 1000, 1500], gamma=0.5)
    elif sched_type == 'exponential':
        scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    loss_f = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    r = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_f, batch_size=64, scheduler=scheduler)
    r.train(train_set, valid_set, num_epochs=3, log_iters=400, save_dir=save_dir)

    m2 = nn.models.Model_MLP()
    m2.load_model(os.path.join(save_dir, 'best_model.pickle'))
    test_acc = nn.metric.accuracy(m2(test_set[0]), test_set[1])
    opt_results[name] = {'best_dev': r.best_score, 'test_acc': test_acc}
    print(f"  Dev: {r.best_score:.4f}, Test: {test_acc:.4f}")

# Optimization bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(opt_results))
w = 0.35
dev_vals = [opt_results[n]['best_dev'] for n in opt_results]
test_vals = [opt_results[n]['test_acc'] for n in opt_results]
ax.bar(x - w/2, dev_vals, w, label='Dev Accuracy', color='#E3E37D')
ax.bar(x + w/2, test_vals, w, label='Test Accuracy', color='#968A62')
ax.set_xticks(x)
ax.set_xticklabels(opt_results.keys(), fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('Optimization Comparison (3 epochs)')
ax.legend()
ax.set_ylim(0.7, 1.0)
for i, (dv, tv) in enumerate(zip(dev_vals, test_vals)):
    ax.text(i - w/2, dv + 0.005, f'{dv:.4f}', ha='center', fontsize=8)
    ax.text(i + w/2, tv + 0.005, f'{tv:.4f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimization_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nOptimization Results:")
print(f"{'Config':<28} {'Dev':<8} {'Test':<8}")
print("-"*44)
for name, r in opt_results.items():
    print(f"{name:<28} {r['best_dev']:<8.4f} {r['test_acc']:<8.4f}")

np.savez(os.path.join(save_dir, 'optimization_results.npz'), results=opt_results)

# ============================================================
# Part C Direction 5: Error Analysis
# ============================================================
print("\n" + "="*60)
print("Part C Direction 5: Error Analysis & Visualization")
print("="*60)

# Load MLP
model = nn.models.Model_MLP()
model.load_model(os.path.join(save_dir, 'mlp_best.pickle'))
logits = model(test_set[0])
preds = np.argmax(logits, axis=1)
acc = nn.metric.accuracy(logits, test_set[1])
print(f"MLP Test Accuracy: {acc:.4f}")

# Confusion Matrix
n_classes = 10
cm = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(test_set[1], preds):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix (Test Accuracy: {acc:.2%})', fontsize=14)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
plt.colorbar(im)
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Per-class accuracy
per_class = np.array([(cm[c,c]/cm[c].sum()) if cm[c].sum()>0 else 0 for c in range(n_classes)])
print("\nPer-class accuracy:")
for c in range(n_classes):
    print(f"  Digit {c}: {per_class[c]:.4f} ({cm[c,c]}/{cm[c].sum()})")

# Misclassified examples
wrong_idx = np.where(preds != test_set[1])[0]
np.random.shuffle(wrong_idx)
n_show = 20
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.reshape(-1)
for i in range(n_show):
    idx = wrong_idx[i]
    axes[i].imshow(test_set[0][idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'True:{test_set[1][idx]} Pred:{preds[idx]}', fontsize=9, color='red')
    axes[i].axis('off')
for i in range(n_show, len(axes)):
    axes[i].axis('off')
plt.suptitle('Misclassified Examples (MLP)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'misclassified.png'), dpi=150, bbox_inches='tight')
plt.close()

# Weight visualization
w1 = model.layers[0].params['W']  # [784, 600]
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.reshape(-1)
vmax = max(abs(w1.min()), abs(w1.max()))
for i in range(25):
    axes[i].imshow(w1[:, i].reshape(28, 28), cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[i].axis('off')
plt.suptitle('MLP First-Layer Weights (25 of 600 hidden units)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()

hardest = np.argsort(per_class)
print(f"\nHardest digits: {list(hardest[:3])}")
print(f"Easiest digits: {list(hardest[-3:][::-1])}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\n{'='*40}")
print(f"Part A - MLP Baseline:")
print(f"  Dev Accuracy:  {results['MLP']['best_dev']:.4f}")
print(f"  Test Accuracy: {results['MLP']['test_acc']:.4f}")
print(f"  Train Time:    {results['MLP']['time']:.1f}s")

print(f"\n{'='*40}")
print(f"Part B - CNN:")
print(f"  Parameters:    {results['CNN']['params']:,} (vs MLP: ~476,000)")
print(f"  Dev Accuracy:  {results['CNN']['best_dev']:.4f}")
print(f"  Test Accuracy: {results['CNN']['test_acc']:.4f}")
print(f"  Train Time:    {results['CNN']['time']:.1f}s")

print(f"\n{'='*40}")
print(f"Part C Direction 1 - Optimization:")
for name, r in opt_results.items():
    print(f"  {name:<28} Dev={r['best_dev']:.4f}  Test={r['test_acc']:.4f}")

print(f"\n{'='*40}")
print(f"Part C Direction 5 - Error Analysis:")
print(f"  MLP Test Accuracy: {acc:.4f}")
print(f"  Hardest: {list(hardest[:3])} (acc: {[f'{per_class[c]:.3f}' for c in hardest[:3]]})")
print(f"  Easiest: {list(hardest[-3:][::-1])} (acc: {[f'{per_class[c]:.3f}' for c in hardest[-3:][::-1]]})")

print(f"\nAll results saved to: {os.path.abspath(save_dir)}/")
print("Generated files:")
for f in sorted(os.listdir(save_dir)):
    size = os.path.getsize(os.path.join(save_dir, f))
    print(f"  {f} ({size:,} bytes)")
