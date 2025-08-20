import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning
from .models.cbm import ConceptBottleneckModel, ConceptModel, LabelPredictor
from .models.cem import ConceptEmbeddingModel

def calculate_task_class_weights(n_tasks, train_dl):
    attribute_count = np.zeros((max(n_tasks, 2),))
    samples_seen = 0
    for _, data in enumerate(train_dl):
        (_, y, _) = data
        if n_tasks > 1:
            y = torch.nn.functional.one_hot(
                y,
                num_classes=n_tasks,
            ).cpu().detach().numpy()
        else:
            y = torch.cat(
                [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                dim=-1,
            ).cpu().detach().numpy()
        attribute_count += np.sum(y, axis=0)
        samples_seen += y.shape[0]
    print("Class distribution is:", attribute_count / samples_seen)
    if n_tasks > 1:
        task_class_weights = samples_seen / attribute_count - 1
    else:
        task_class_weights = np.array(
            [attribute_count[0]/attribute_count[1]]
        )

    return torch.tensor(task_class_weights, dtype=torch.float32)

def calculate_concept_loss_weights(n_concepts, train_dl):
    attribute_count = np.zeros((n_concepts,))
    samples_seen = 0
    for _, data in enumerate(train_dl):
        (_, _, c) = data
        c = c.cpu().detach().numpy()
        c = np.nan_to_num(c)
        attribute_count += np.sum(c, axis=0)
        samples_seen += c.shape[0]
    attribute_count[attribute_count == 0] = 1
    imbalance = samples_seen / attribute_count - 1

    return torch.tensor(imbalance, dtype=torch.float32)

def train_cbm_joint(
        datasets,
        save_path=None,
        max_epochs=300,
        use_task_class_weights=False,
        use_concept_loss_weights=False):
    task_class_weights = None
    concept_loss_weights = None
    if use_task_class_weights:
        task_class_weights = calculate_task_class_weights(datasets.n_tasks, datasets.train_dl())
    if use_concept_loss_weights:
        concept_loss_weights = calculate_concept_loss_weights(datasets.n_concepts, datasets.train_dl())

    model = ConceptBottleneckModel(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        latent_representation_size=datasets.latent_representation_size,
        task_class_weights=task_class_weights,
        concept_loss_weights=concept_loss_weights,
        concept_names=datasets.concept_names,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, datasets.train_dl(), datasets.val_dl())

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, datasets.test_dl())
    [recipeless_test_results] = trainer.test(model, datasets.recipeless_test_dl())

    return model, test_results, recipeless_test_results

def train_cbm_sequential(
        datasets,
        use_recipeless=False,
        save_path=None,
        max_epochs=300,
        use_task_class_weights=False,
        use_concept_loss_weights=False):
    task_class_weights = None
    concept_loss_weights = None
    if use_task_class_weights:
        task_class_weights = calculate_task_class_weights(datasets.n_tasks, datasets.train_dl())
    if use_concept_loss_weights:
        concept_loss_weights = calculate_concept_loss_weights(datasets.n_concepts, datasets.train_dl())

    concept_model = ConceptModel(
        latent_representation_size=datasets.latent_representation_size,
        n_concepts=datasets.n_concepts,
        concept_loss_weights=concept_loss_weights,
        concept_names=datasets.concept_names
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5
    )

    if use_recipeless:
        trainer.fit(concept_model, datasets.recipeless_train_dl(), datasets.recipeless_val_dl())
    else:
        trainer.fit(concept_model, datasets.train_dl(), datasets.val_dl())

    concept_model.eval()
    concept_model.freeze()

    def predict_concepts(dataloader):
        all_examples, all_labels, all_concepts = [], [], []
        for x, y, _ in dataloader:
            with torch.no_grad():
                preds = concept_model(x)
            all_examples.append(x.cpu())
            all_labels.append(y.cpu())
            all_concepts.append(preds.cpu())
        return torch.cat(all_examples), torch.cat(all_labels), torch.cat(all_concepts)

    train_x, train_y, train_c = predict_concepts(datasets.train_dl())
    val_x, val_y, val_c = predict_concepts(datasets.val_dl())

    train_label_dl = DataLoader(TensorDataset(train_x, train_y, train_c), batch_size=256, num_workers=7)
    val_label_dl = DataLoader(TensorDataset(val_x, val_y, val_c), batch_size=256, num_workers=7)

    label_predictor = LabelPredictor(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        task_class_weights=task_class_weights
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5
    )
    trainer.fit(label_predictor, train_label_dl, val_label_dl)

    model = ConceptBottleneckModel(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        latent_representation_size=datasets.latent_representation_size,
        task_class_weights=task_class_weights,
        concept_loss_weights=concept_loss_weights,
        concept_names=datasets.concept_names,
    )

    model.concept_model.load_state_dict(concept_model.concept_model.state_dict())
    model.label_predictor.load_state_dict(label_predictor.label_predictor.state_dict())

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, datasets.test_dl())
    [recipeless_test_results] = trainer.test(model, datasets.recipeless_test_dl())

    return model, test_results, recipeless_test_results

def train_cbm_independent(
        datasets,
        use_recipeless=False,
        save_path=None,
        max_epochs=300,
        use_task_class_weights=False,
        use_concept_loss_weights=False):
    task_class_weights = None
    concept_loss_weights = None
    if use_task_class_weights:
        task_class_weights = calculate_task_class_weights(datasets.n_tasks, datasets.train_dl())
    if use_concept_loss_weights:
        concept_loss_weights = calculate_concept_loss_weights(datasets.n_concepts, datasets.train_dl())

    concept_model = ConceptModel(
        latent_representation_size=datasets.latent_representation_size,
        n_concepts=datasets.n_concepts,
        concept_loss_weights=concept_loss_weights,
        concept_names=datasets.concept_names
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5
    )

    if use_recipeless:
        trainer.fit(concept_model, datasets.recipeless_train_dl(), datasets.recipeless_val_dl())
    else:
        trainer.fit(concept_model, datasets.train_dl(), datasets.val_dl())

    label_predictor = LabelPredictor(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        task_class_weights=task_class_weights
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5
    )
    trainer.fit(label_predictor, datasets.train_dl(), datasets.val_dl())

    model = ConceptBottleneckModel(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        latent_representation_size=datasets.latent_representation_size,
        task_class_weights=task_class_weights,
        concept_loss_weights=concept_loss_weights,
        concept_names=datasets.concept_names,
    )

    model.concept_model.load_state_dict(concept_model.concept_model.state_dict())
    model.label_predictor.load_state_dict(label_predictor.label_predictor.state_dict())

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, datasets.test_dl())
    [recipeless_test_results] = trainer.test(model, datasets.recipeless_test_dl())

    return model, test_results, recipeless_test_results

def train_cem(
        datasets,
        save_path=None,
        max_epochs=300,
        use_task_class_weights=False,
        use_concept_loss_weights=False):
    task_class_weights = None
    concept_loss_weights = None
    if use_task_class_weights:
        task_class_weights = calculate_task_class_weights(datasets.n_tasks, datasets.train_dl())
    if use_concept_loss_weights:
        concept_loss_weights = calculate_concept_loss_weights(datasets.n_concepts, datasets.train_dl())

    model = ConceptEmbeddingModel(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        pre_concept_model=None,
        latent_representation_size=datasets.latent_representation_size,
        task_class_weights=task_class_weights,
        concept_names=datasets.concept_names,
        concept_loss_weights=concept_loss_weights
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, datasets.train_dl(), datasets.val_dl())

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, datasets.test_dl())
    [recipeless_test_results] = trainer.test(model, datasets.recipeless_test_dl())

    return model, test_results, recipeless_test_results

def train_black_box(
        datasets,
        save_path=None,
        max_epochs=300,
        use_task_class_weights=False):
    task_class_weights = None
    if use_task_class_weights:
        task_class_weights = calculate_task_class_weights(datasets.n_tasks, datasets.train_dl())

    model = ConceptEmbeddingModel(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        pre_concept_model=None,
        latent_representation_size=datasets.latent_representation_size,
        task_class_weights=task_class_weights,
        concept_loss_weights=None,
        concept_names=datasets.concept_names,
        concept_loss_weight=0
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, datasets.train_dl(), datasets.val_dl())

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, datasets.test_dl())

    return model, test_results
