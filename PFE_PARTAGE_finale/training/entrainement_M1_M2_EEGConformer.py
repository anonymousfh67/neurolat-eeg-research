# =========================================================
# 4. COLLECTE ET PRÉPARATION DU JEU DE DONNÉES M1 (Normal/Crise)
# =========================================================
print("\n" + "="*60)
print("📦 PHASE 1 : Préparation du Dataset M1 (Normal vs Crise)")
print("="*60)

all_X_m1 = {'normal': [], 'crise': []}
csvs = glob.glob(f'{TUSZ_TRAIN}/**/*.csv', recursive=True)
random.seed(42)
random.shuffle(csvs)

SEIZURE_LABELS = ['fnsz','gnsz','cpsz','spsz','tnsz','tcsz','absz','mysz']

def extraire_intervalles_crise(csv_path):
    try:
        df = pd.read_csv(csv_path, comment='#', header=None)
        df.columns = ['ch','start','stop','label','conf']
        df = df[df['ch'] != 'ch']
        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['stop']  = pd.to_numeric(df['stop'], errors='coerce')
        df = df.dropna(subset=['start','stop'])
    except: return []
    df_sz = df[df['label'].isin(SEIZURE_LABELS)]
    if df_sz.empty: return []
    intervals = sorted(df_sz[['start','stop']].values.tolist())
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def fenetre_label_m1(t_start, t_stop, intervalles, seuil=0.5):
    duree = t_stop - t_start
    overlap = sum(max(0, min(t_stop, e) - max(t_start, s)) for s, e in intervalles)
    frac = overlap / duree
    if frac >= seuil: return 'crise'
    if frac == 0.0: return 'normal'
    return None

traites = 0
for csv_path in csvs:
    if all(len(v) >= N_TARGET_M1 for v in all_X_m1.values()):
        break
    edf_path = csv_path.replace('.csv', '.edf')
    if not os.path.exists(edf_path): continue
    intervalles = extraire_intervalles_crise(csv_path)
    data = pretraiter_edf_v3(edf_path)
    if data is None: continue

    n_total = data.shape[1]
    pos = 0
    while pos + WIN_SAMP <= n_total:
        t_start, t_stop = pos / FS, (pos + WIN_SAMP) / FS
        lbl = fenetre_label_m1(t_start, t_stop, intervalles)
        if lbl == 'crise' and len(all_X_m1['crise']) < N_TARGET_M1:
            all_X_m1['crise'].append(normaliser_segment(data[:, pos:pos+WIN_SAMP].copy()))
            pos += WIN_SAMP // 2
        elif lbl == 'normal' and len(all_X_m1['normal']) < N_TARGET_M1:
            all_X_m1['normal'].append(normaliser_segment(data[:, pos:pos+WIN_SAMP].copy()))
            pos += WIN_SAMP
        else:
            pos += WIN_SAMP
    traites += 1

# Équilibrage
n_min_m1 = min(len(v) for v in all_X_m1.values())
for cls in all_X_m1:
    random.shuffle(all_X_m1[cls])
    all_X_m1[cls] = all_X_m1[cls][:n_min_m1]

# Split 80/10/10
X_tr1, y_tr1, X_dv1, y_dv1, X_ts1, y_ts1 = [], [], [], [], [], []
for cls, segs in all_X_m1.items():
    n = len(segs)
    n_tr, n_dv = int(n * 0.8), int(n * 0.1)
    X_tr1 += segs[:n_tr]; y_tr1 += [cls] * n_tr
    X_dv1 += segs[n_tr:n_tr+n_dv]; y_dv1 += [cls] * n_dv
    X_ts1 += segs[n_tr+n_dv:]; y_ts1 += [cls] * (n - n_tr - n_dv)

# Shuffling et conversion
idx_tr1 = list(range(len(X_tr1)))
random.shuffle(idx_tr1)
X_tr1 = np.array([X_tr1[i] for i in idx_tr1], dtype=np.float32)
y_tr1 = np.array([0 if y_tr1[i]=='normal' else 1 for i in idx_tr1], dtype=np.int64)

X_dv1 = np.array(X_dv1, dtype=np.float32)
y_dv1 = np.array([0 if y=='normal' else 1 for y in y_dv1], dtype=np.int64)
X_ts1 = np.array(X_ts1, dtype=np.float32)
y_ts1 = np.array([0 if y=='normal' else 1 for y in y_ts1], dtype=np.int64)

# Normalisation globale
mu1 = X_tr1.mean()
std1 = X_tr1.std() + 1e-8
X_tr1 = (X_tr1 - mu1) / std1
X_dv1 = (X_dv1 - mu1) / std1
X_ts1 = (X_ts1 - mu1) / std1

np.save(f'{MODELS}/mu1_v2.npy', np.array([mu1]))
np.save(f'{MODELS}/std1_v2.npy', np.array([std1]))

print(f"[M1] Train: {X_tr1.shape}, Dev: {X_dv1.shape}, Test: {X_ts1.shape}")

# =========================================================
# 5. ENTRAÎNEMENT DU MODÈLE M1 (Normal/Crise)
# =========================================================
print("\n" + "="*60)
print("🚀 PHASE 2 : Entraînement du Modèle M1")
print("="*60)

train_loader1 = DataLoader(TensorDataset(torch.tensor(X_tr1), torch.tensor(y_tr1)), batch_size=BATCH_SIZE, shuffle=True)
dev_loader1   = DataLoader(TensorDataset(torch.tensor(X_dv1), torch.tensor(y_dv1)), batch_size=BATCH_SIZE, shuffle=False)

model_m1 = EEGConformer(n_classes=2).to(device)
criterion1 = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer1 = torch.optim.Adam(model_m1.parameters(), lr=0.0001)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=N_EPOCHS, eta_min=1e-6)

best_f1_m1, patience_cpt1 = 0.0, 0
for epoch in range(N_EPOCHS):
    model_m1.train()
    tr_loss = 0.0
    for X_b, y_b in train_loader1:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer1.zero_grad()
        loss = criterion1(model_m1(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model_m1.parameters(), 1.0)
        optimizer1.step()
        tr_loss += loss.item()
    scheduler1.step()

    model_m1.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in dev_loader1:
            out = model_m1(X_b.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y_b.numpy())

    dv_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
    if dv_f1 > best_f1_m1:
        best_f1_m1, patience_cpt1 = dv_f1, 0
        torch.save(model_m1.state_dict(), f'{MODELS}/model_m1_v2.pt')
        print(f"   💾 Ep {epoch+1:2d} - Best F1 M1={best_f1_m1:.4f}")
    else:
        patience_cpt1 += 1
        if patience_cpt1 >= PATIENCE:
            print("⏹️ Early Stopping M1!")
            break

# =========================================================
# 6. COLLECTE ET PRÉPARATION DU JEU DE DONNÉES M2 (ILAE)
# =========================================================
print("\n" + "="*60)
print("📦 PHASE 3 : Préparation du Dataset M2 (Gauche vs Droite vs Bilatéral)")
print("="*60)

all_X_m2 = {'gauche': [], 'droite': [], 'bilateral': []}
random.shuffle(csvs)

for csv_path in csvs:
    if all(len(v) >= N_TARGET_M2 for v in all_X_m2.values()):
        break
    edf_path = csv_path.replace('.csv', '.edf')
    if not os.path.exists(edf_path): continue
    events = extraire_evenements_m2(csv_path)
    if not events: continue
    data = pretraiter_edf_v3(edf_path)
    if data is None: continue

    for ev in events:
        label = ev['label']
        if len(all_X_m2[label]) >= N_TARGET_M2: continue
        start_s = int(ev['start'] * FS)
        stop_s  = min(int(ev['stop'] * FS), data.shape[1])
        step, pos = WIN_SAMP // 2, start_s
        while pos + WIN_SAMP <= stop_s:
            if len(all_X_m2[label]) >= N_TARGET_M2: break
            all_X_m2[label].append(normaliser_segment(data[:, pos:pos+WIN_SAMP].copy()))
            pos += step

n_min_m2 = min(len(v) for v in all_X_m2.values())
for cls in all_X_m2:
    random.shuffle(all_X_m2[cls])
    all_X_m2[cls] = all_X_m2[cls][:n_min_m2]

# Split 80/10/10
X_tr2, y_tr2, X_dv2, y_dv2, X_ts2, y_ts2 = [], [], [], [], [], []
for cls, segs in all_X_m2.items():
    n = len(segs)
    n_tr, n_dv = int(n * 0.8), int(n * 0.1)
    X_tr2 += segs[:n_tr]; y_tr2 += [cls] * n_tr
    X_dv2 += segs[n_tr:n_tr+n_dv]; y_dv2 += [cls] * n_dv
    X_ts2 += segs[n_tr+n_dv:]; y_ts2 += [cls] * (n - n_tr - n_dv)

# Shuffling et conversion
idx_tr2 = list(range(len(X_tr2)))
random.shuffle(idx_tr2)
X_tr2 = np.array([X_tr2[i] for i in idx_tr2], dtype=np.float32)
label2idx_m2 = {c: i for i, c in enumerate(CLASSES_M2)}
y_tr2 = np.array([label2idx_m2[y_tr2[i]] for i in idx_tr2], dtype=np.int64)

X_dv2 = np.array(X_dv2, dtype=np.float32)
y_dv2 = np.array([label2idx_m2[y] for y in y_dv2], dtype=np.int64)
X_ts2 = np.array(X_ts2, dtype=np.float32)
y_ts2 = np.array([label2idx_m2[y] for y in y_ts2], dtype=np.int64)

# Normalisation globale
mu2 = X_tr2.mean()
std2 = X_tr2.std() + 1e-8
X_tr2 = (X_tr2 - mu2) / std2
X_dv2 = (X_dv2 - mu2) / std2
X_ts2 = (X_ts2 - mu2) / std2

np.save(f'{MODELS}/mu2_v3.npy', np.array([mu2]))
np.save(f'{MODELS}/std2_v3.npy', np.array([std2]))

print(f"[M2] Train: {X_tr2.shape}, Dev: {X_dv2.shape}, Test: {X_ts2.shape}")

# =========================================================
# 7. ENTRAÎNEMENT DU MODÈLE M2 (G/D/Bilateral)
# =========================================================
print("\n" + "="*60)
print("🚀 PHASE 4 : Entraînement du Modèle M2")
print("="*60)

train_loader2 = DataLoader(TensorDataset(torch.tensor(X_tr2), torch.tensor(y_tr2)), batch_size=BATCH_SIZE, shuffle=True)
dev_loader2   = DataLoader(TensorDataset(torch.tensor(X_dv2), torch.tensor(y_dv2)), batch_size=BATCH_SIZE, shuffle=False)

model_m2 = EEGConformer(n_classes=3).to(device)
criterion2 = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer2 = torch.optim.Adam(model_m2.parameters(), lr=0.0001)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=N_EPOCHS, eta_min=1e-6)

best_f1_m2, patience_cpt2 = 0.0, 0
for epoch in range(N_EPOCHS):
    model_m2.train()
    tr_loss = 0.0
    for X_b, y_b in train_loader2:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer2.zero_grad()
        loss = criterion2(model_m2(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model_m2.parameters(), 1.0)
        optimizer2.step()
        tr_loss += loss.item()
    scheduler2.step()

    model_m2.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in dev_loader2:
            out = model_m2(X_b.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y_b.numpy())

    dv_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
    if dv_f1 > best_f1_m2:
        best_f1_m2, patience_cpt2 = dv_f1, 0
        torch.save(model_m2.state_dict(), f'{MODELS}/model_m2_v3.pt')
        print(f"   💾 Ep {epoch+1:2d} - Best F1 M2={best_f1_m2:.4f}")
    else:
        patience_cpt2 += 1
        if patience_cpt2 >= PATIENCE:
            print("⏹️ Early Stopping M2!")
            break

# =========================================================
# 8. VALIDATION ET RAPPORTS FINAUX SUR LE JEU DE TEST
# =========================================================
print("\n" + "="*60)
print("🏆 PHASE 5 : Évaluation Finale sur Test")
print("="*60)

# Test M1
model_m1.load_state_dict(torch.load(f'{MODELS}/model_m1_v2.pt', map_location=device))
model_m1.eval()
test_preds1 = []
with torch.no_grad():
    for i in range(0, len(X_ts1), BATCH_SIZE):
        out = model_m1(torch.tensor(X_ts1[i:i+BATCH_SIZE]).to(device))
        test_preds1.extend(out.argmax(1).cpu().numpy())

print("\n📊 RAPPORT M1 (Normal vs Crise) :")
print(classification_report(y_ts1, test_preds1, target_names=CLASSES_M1, zero_division=0))

# Test M2
model_m2.load_state_dict(torch.load(f'{MODELS}/model_m2_v3.pt', map_location=device))
model_m2.eval()
test_preds2 = []
with torch.no_grad():
    for i in range(0, len(X_ts2), BATCH_SIZE):
        out = model_m2(torch.tensor(X_ts2[i:i+BATCH_SIZE]).to(device))
        test_preds2.extend(out.argmax(1).cpu().numpy())

print("\n📊 RAPPORT M2 (Gauche vs Droite vs Bilatéral) :")
print(classification_report(y_ts2, test_preds2, target_names=CLASSES_M2, zero_division=0))
