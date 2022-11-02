import torch.utils.data
from tqdm import tqdm
from config import *
from Effmodel import EffnetModel
from dataset import EffnetDataSet
import os
from loss import weighted_loss
import wandb


def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])
#%%
def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.tph')
#%%
def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data)
    return model


# quick test
model = torch.nn.Linear(2, 1)
save_model('testmodel', model)

model1 = load_model(torch.nn.Linear(2, 1), 'testmodel')
assert torch.all(
    next(iter(model1.parameters())) == next(iter(model.parameters()))
).item(), "Loading/saving is inconsistent!"
#%%
def evaluate_effnet(model: EffnetModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(),
                                          collate_fn=filter_nones)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []
        vert_losses = []
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                with autocast():
                    y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                    frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE)).item()
                    vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE)).item()
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    pred_vert.append(torch.sigmoid(y_vert_pred))
                    frac_losses.append(frac_loss)
                    vert_losses.append(vert_loss)

                if i >= max_batches:
                    break
        return np.mean(frac_losses), np.mean(vert_losses), torch.concat(pred_frac).cpu().numpy(), torch.concat(pred_vert).cpu().numpy()


def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


def train_effnet(ds_train, ds_eval, logger, name):
    torch.manual_seed(42)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(),
                                           collate_fn=filter_nones)

    model = EffnetModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=ONE_CYCLE_MAX_LR, epochs=1,
                                                    steps_per_epoch=min(EFFNET_MAX_TRAIN_BATCHES, len(dl_train)),
                                                    pct_start=ONE_CYCLE_PCT_START)

    model.train()
    scaler = GradScaler()
    for batch_idx, (X, y_frac, y_vert) in enumerate(tqdm(dl_train, desc='Train', miniters=10)):

        if ds_eval is not None and batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0 and EFFNET_MAX_EVAL_BATCHES > 0:
            frac_loss, vert_loss = evaluate_effnet(
                model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)[:2]
            model.train()
            logger.log(
                {'eval_frac_loss': frac_loss, 'eval_vert_loss': vert_loss, 'eval_loss': frac_loss + vert_loss})
            if batch_idx > 0:  # don't save untrained model
                save_model(name, model)

        if batch_idx >= EFFNET_MAX_TRAIN_BATCHES:
            break

        optim.zero_grad()
        # Using mixed precision training
        with autocast():
            y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
            frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE))
            vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE))
            loss = FRAC_LOSS_WEIGHT * frac_loss + vert_loss

            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f'Bad loss, skipping the batch {batch_idx}')
                del loss, frac_loss, vert_loss, y_frac_pred, y_vert_pred
                gc_collect()
                continue

        # scaler is needed to prevent "gradient underflow"
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        # progress.set_description(f'Train loss: {loss.item() :.02f}')
        logger.log({'loss': (loss.item()), 'frac_loss': frac_loss.item(), 'vert_loss': vert_loss.item(),
                    'lr': scheduler.get_last_lr()[0]})

    save_model(name, model)
    return model

if __name__ == '__main__':
    df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')
    df_train.sample(2)
    df_train_slices = pd.read_csv(f'{METADATA_PATH}/train_segmented.csv')
    c1c7 = [f'C{i}' for i in range(1, 8)]
    df_train_slices[c1c7] = (df_train_slices[c1c7] > 0.5).astype(int)
    #%%
    df_train = df_train_slices.set_index('StudyInstanceUID').join(df_train.set_index('StudyInstanceUID'),
                                                                  rsuffix='_fracture').reset_index().copy()
    df_train = df_train.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)
    df_train.sample(2)
    split = GroupKFold(N_FOLDS)
    for k, (_, test_idx) in enumerate(split.split(df_train, groups=df_train.StudyInstanceUID)):
        df_train.loc[test_idx, 'split'] = k
    df_train.sample(2)

    df_test = pd.read_csv(f'{RSNA_2022_PATH}/test.csv')

    if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
        # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
        df_test = pd.DataFrame({
            "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
            "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
            "prediction_type": ["C1", "C1", "patient_overall"]}
        )

    # N-fold models. Can be used to estimate accurate CV score and in ensembled submissions.
    effnet_models = []
    for fold in range(1):
        if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'effnetv2-f{fold}.tph')):
            print(f'Found cached version of effnetv2-f{fold}')
            effnet_models.append(load_model(EffnetModel(), f'effnetv2-f{fold}', EFFNET_CHECKPOINTS_PATH))
        else:
            with wandb.init(project='RSNA-2022', name=f'EffNet-v2-fold{fold}') as run:
                wandb.init(project='RSNA-2022', name=f'EffNet-v2-fold{fold}')
                gc_collect()
                ds_train = EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                ds_eval = EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                effnet_models.append(train_effnet(ds_train, ds_eval, run, f'effnetv2-f{fold}'))
    # "Main" model that uses all folds data. Can be used in single-model submissions.
    # if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'effnetv2.tph')):
    #     print(f'Found cached version of effnetv2')
    #     effnet_models.append(load_model(EffnetModel(), f'effnetv2', EFFNET_CHECKPOINTS_PATH))
    # else:
    #     with wandb.init(project='RSNA-2022', name=f'EffNet-v2') as run:
    #         gc_collect()
    #         ds_train = EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
    #         train_effnet(ds_train, None, run, f'effnetv2')