import torch
import lightning
import numpy as np
from . import base
from ..metrics import calculate_concept_accuracies, calculate_task_accuracy

class ConceptBottleneckModel(base.BaseModel):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            latent_representation_size,
            task_class_weights,
            concept_loss_weights,
            concept_names):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights, concept_names)
        self.n_concepts = n_concepts
        self.concept_loss_weight = 10

        # Representations from the foundation model are precomputed and passed in the dataset.
        self.concept_model = torch.nn.Linear(latent_representation_size, self.n_concepts)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, c_true=None, train=False):
        predicted_concept_logits = self.concept_model(x)
        predicted_concept_probs = self.sigmoid(predicted_concept_logits)

        interventions = None
        if self.intervention_mask is not None:
            interventions = torch.tile(self.intervention_mask, (predicted_concept_probs.shape[0], 1))

        if c_true is not None and interventions is not None:
            interventions = interventions.to(predicted_concept_probs.device)
            if isinstance(self.intervention_off_value, torch.Tensor):
                intervention_off_value = self.intervention_off_value.to(
                    dtype=torch.float32,
                    device=predicted_concept_probs.device)
            else:
                intervention_off_value = self.intervention_off_value
            if isinstance(self.intervention_on_value, torch.Tensor):
                intervention_on_value = self.intervention_on_value.to(
                    dtype=torch.float32,
                    device=predicted_concept_probs.device)
            else:
                intervention_on_value = self.intervention_on_value

            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                torch.where(c_true == 0, intervention_off_value, intervention_on_value),
                predicted_concept_probs
            )

            concept_probs_after_interventions = predicted_concept_probs * (1 - interventions) + interventions * c_true
        else:
            concept_probs_after_interventions = predicted_concept_probs

        predicted_labels = self.label_predictor(concept_probs_after_interventions)

        return predicted_concept_probs, predicted_labels

class ConceptModel(lightning.LightningModule):
    def __init__(self, latent_representation_size, n_concepts, concept_loss_weights, concept_names):
        super().__init__()

        self.concept_model = torch.nn.Linear(latent_representation_size, n_concepts)
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=concept_loss_weights)

        self.concept_names = concept_names

    def forward(self, x):
        predicted_concept_logits = self.concept_model(x)
        return self.sigmoid(predicted_concept_logits)

    def run_step(self, batch):
        x, _, c = batch

        predicted_concept_probs = self.forward(x)

        concept_loss = 0
        c_accuracy, c_accuracies, c_auc, c_aucs = np.nan, [np.nan], np.nan, [np.nan]
        c_used = torch.where(
            torch.logical_and(c >= 0, c <= 1),
            c,
            torch.zeros_like(c)
        )
        predicted_concept_probs_used = torch.where(
            torch.logical_and(c >= 0, c <= 1),
            predicted_concept_probs,
            torch.zeros_like(predicted_concept_probs)
        )

        concept_loss = self.loss_concept(predicted_concept_probs_used, c_used)
        c_accuracy, c_accuracies, c_auc, c_aucs = calculate_concept_accuracies(predicted_concept_probs, c)

        result = {
            "c_accuracy": c_accuracy,
            "c_accuracies": c_accuracies,
            "c_auc": c_auc,
            "c_aucs": c_aucs,
        }

        return concept_loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("loss", float(loss), prog_bar=True)
        self.log("c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("c_auc", result["c_auc"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"{self.concept_names[i]}_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"{self.concept_names[i]}_auc", auc)
        return {
            "loss": loss,
            "log": {**result, "loss": float(loss)}
        }

    def validation_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("val_loss", float(loss), prog_bar=True)
        self.log("val_c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("val_c_auc", result["c_auc"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"{self.concept_names[i]}_val_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"{self.concept_names[i]}_val_auc", auc)
        return {
            "val_" + key: val for key, val in list(result.items()) + [("loss", float(loss))]
        }

    def test_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("test_loss", float(loss), prog_bar=True)
        self.log("test_c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("test_c_auc", result["c_auc"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"{self.concept_names[i]}_test_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"{self.concept_names[i]}_test_auc", auc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

class LabelPredictor(lightning.LightningModule):
    def __init__(self, n_concepts, n_tasks, task_class_weights):
        super().__init__()

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, n_tasks)
        )

        if n_tasks > 1:
            self.loss_task = torch.nn.CrossEntropyLoss(weight=task_class_weights)
        else:
            self.loss_task = torch.nn.BCEWithLogitsLoss(weight=task_class_weights)

    def forward(self, x):
        return self.label_predictor(x)

    def run_step(self, batch):
        _, y, c = batch

        predicted_labels = self.forward(c)

        task_loss = self.loss_task(predicted_labels.squeeze(), y)

        y_accuracy = calculate_task_accuracy(predicted_labels, y)

        result = {
            "y_accuracy": y_accuracy
        }

        return task_loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("loss", float(loss), prog_bar=True)
        self.log("y_accuracy", result["y_accuracy"], prog_bar=True)
        return {
            "loss": loss,
            "log": {**result, "loss": float(loss)}
        }

    def validation_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("val_loss", float(loss), prog_bar=True)
        self.log("val_y_accuracy", result["y_accuracy"], prog_bar=True)
        return {
            "val_" + key: val for key, val in list(result.items()) + [("loss", float(loss))]
        }

    def test_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("test_loss", float(loss), prog_bar=True)
        self.log("test_y_accuracy", result["y_accuracy"], prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
