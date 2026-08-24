import os
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Model
import numpy as np
from transformers import get_linear_schedule_with_warmup

from model.whisper_aasist import (
    GraphAttentionLayer,
    HtrgGraphAttentionLayer,
    Residual_block,
    GraphPool,
)

counter = 0


class Wav2Vec2(nn.Module):
    @property
    def out_dim(self):
        return self.model.config.hidden_size

    def __init__(self, device: str = "cuda"):
        super().__init__()

        ckpt_path = "facebook/wav2vec2-xls-r-300m"
        self.device = device

        model = Wav2Vec2Model.from_pretrained(ckpt_path)
        model.encoder.layers = model.encoder.layers[:16]
        self.model = apply_svd_residual_to_self_attn(model, 120)


    def forward(self, x: torch.Tensor, attention_mask = None, sample_rate: int = 16_000):
        x = self.model(x, attention_mask=attention_mask).last_hidden_state
        return x


class W2V_AASIST(nn.Module):
    def __init__(self, device: str, num_classes: int = 1):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = Wav2Vec2(device=self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], self.num_classes)

        ####
        # instantiate loss and optimizer
        ####
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=4e-5, weight_decay=4e-6)

    def forward(self, x: torch.Tensor, attention_mask = None ) -> torch.Tensor:
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model(x.squeeze(-1), attention_mask=attention_mask)
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # -------pre-trained Wav2vec model fine tunning ------------------------##

        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)

        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)

        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)

        # ------------SAP for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)

        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)

        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SAP for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        # print(last_hidden.amax(dim=1), last_hidden.amin(dim=1), last_hidden.mean(dim=1))
        last_hidden = self.drop(last_hidden)

        output = self.out_layer(last_hidden)

        return output, last_hidden

    def start_train(
        self, train_loader, test_loader, ood_loader, checkpoint_path, epochs=10
    ):
        # Add learning rate decay scheduler + warmup
        num_train_steps = len(train_loader) * epochs
        num_warmup_steps = int(0.2 * num_train_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            print("Evaluate on train dataset:")
            train_acc, train_roc_auc, train_eer, best_thr_train, best_acc_train = self.evaluate(
                train_loader, verbose=False
            )
            print("Evaluate on validation dataset:")
            val_acc, val_roc_auc, val_eer, best_thr_val, best_acc_val = self.evaluate(
                test_loader, verbose=False
            )
            print("Evaluate on OOD dataset:")
            ood_acc, ood_roc_auc, ood_eer, best_thr_ood, best_acc_ood = self.evaluate(
                ood_loader, verbose=False
            )
            print(
                f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f} - Train[Acc: {train_acc:.4f}%, AUC: {train_roc_auc:.4f}, EER: {train_eer:.4f}] - Val[Acc: {val_acc:.4f}%, AUC: {val_roc_auc:.2f}, EER: {val_eer:.4f}] - OOD[Acc: {ood_acc:.4f}%, AUC: {ood_roc_auc:.4f}, EER: {ood_eer:.4f}, Best thr: {best_thr_ood}, Best acc: {best_acc_ood}]"
            )

            self.save_model(checkpoint_path, epoch)

    def train_epoch(self, loader):
        self.train()
        total_loss = 0.0

        len = 0

        for _, (X_batch, mask_batch, y_batch) in enumerate(tqdm(loader)):
            X_batch, mask_batch, y_batch = (
                X_batch.to(self.device),
                mask_batch.to(self.device),
                y_batch.to(self.device),
            )

            self.optimizer.zero_grad()
            outputs = self(X_batch, attention_mask=mask_batch)
            loss = self.criterion(outputs.squeeze(-1), y_batch.float())
            loss.backward()
            self.optimizer.step()
            #
            self.scheduler.step()
            #
            total_loss += loss.item()
            len += 1
        avg_loss = total_loss / len
        return avg_loss

    def evaluate(self, loader, verbose=True):
        self.eval()

        all_scores = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for _, (X_batch, mask_batch, y_batch) in enumerate(tqdm(loader)):
                X_batch, mask_batch, y_batch = (
                    X_batch.to(self.device),
                    mask_batch.to(self.device),
                    y_batch.to(self.device),
                )
                logits = self.forward(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_labels.extend(y_batch.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        all_preds = np.array(all_preds).reshape(-1)
        all_labels = np.array(all_labels).reshape(-1)
        all_scores = np.array(all_scores).reshape(-1)

        # print(f"Score Range: {all_scores.min():.6f} to {all_scores.max():.6f}")
        # print(f"Mean Score for Fake: {all_scores[all_labels==1].mean():.6f}")
        # print(f"Mean Score for Valid: {all_scores[all_labels==0].mean():.6f}")

        acc = 100 * ((all_preds == all_labels).sum() / len(all_preds))
        roc_auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

        best_thr, best_acc = self.compute_best_thr(thresholds, all_scores, all_labels)

        # Compute EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
        return acc, roc_auc, eer, best_thr, best_acc

    def compute_best_thr(self, thresholds, y_probs, y_true):
        accuracies = []
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred))

        best_threshold = thresholds[np.argmax(accuracies)]
        return best_threshold, np.max(accuracies)

    def predict(self, input_features):
        self.eval()
        with torch.no_grad():
            input_features = input_features.to(self.device)
            logits = self.forward(input_features)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def save_model(self, path, epoch):
        path = path.split(".")[0] + f"_{epoch}.pth"
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f"Model loaded from {path}")

    def freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return (
                self.weight_main
                + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            )
        else:
            return self.weight_main

    def forward(self, x):
        if (
            hasattr(self, "U_residual")
            and hasattr(self, "V_residual")
            and self.S_residual is not None
        ):
            # Reconstruct the residual weight
            residual_weight = (
                self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            )
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = (
                torch.cat((self.U_r, self.U_residual), dim=1)
                @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            )
            VVT = (
                torch.cat((self.V_r, self.V_residual), dim=0)
                @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            )
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()

            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)

            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p="fro") + 0.5 * torch.norm(
                VVT - VVT_identity, p="fro"
            )
        else:
            loss = 0.0

        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = (
                self.weight_main
                + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            )
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p="fro")

            loss = torch.abs(weight_current_fnorm**2 - self.weight_original_fnorm**2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
        else:
            loss = 0.0

        return loss

    def compute_fn_loss(self):
        if self.S_residual is not None:
            weight_current = (
                self.weight_main
                + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            )
            weight_current_fnorm = torch.norm(weight_current, p="fro")

            loss = weight_current_fnorm**2
        else:
            loss = 0.0

        return loss


# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r, n_last=2):
    global counter
    for name, module in reversed(list(model.named_children())):
        if counter >= n_last:
            break
        # for name, module in self_attn_modules[-n_last:]:
        # if "self_attn" in name:
        if "attention" in name:
            has_Linear = False
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        replace_with_svd_residual(sub_module, r),
                    )
                    has_Linear = True
            if has_Linear:
                counter = counter + 1

        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ["S_residual", "U_residual", "V_residual"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def unfreeze_residual_components(model):
    for name, module in reversed(list(model.named_children())):
        unfreeze_residual_components(module)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ["S_residual", "U_residual", "V_residual"]):
            param.requires_grad = True


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(
            in_features,
            out_features,
            r,
            bias=bias,
            init_weight=module.weight.data.clone(),
        )

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p="fro")

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]  # Shape: (out_features, r)
        S_r = S[:r]  # Shape: (r,)
        Vh_r = Vh[:r, :]  # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p="fro")

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]  # Shape: (out_features, n - r)
        S_residual = S[r:]  # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module
