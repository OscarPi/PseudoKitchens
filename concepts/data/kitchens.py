from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import OpenEXR
import struct
import json
from . import transforms
from .base import Datasets, CEMDataset

def hex_str_to_id(id_hex_string):
    packed = struct.Struct("=I").pack(int(id_hex_string, 16))
    return struct.Struct("=f").unpack(packed)[0]

def image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
    for i in range(1, dataset_info["object_counts"]["ingredient_counts"][ingredient] + 1):
        if f"{ingredient} {i}" in manifest:
            id = hex_str_to_id(manifest[f"{ingredient} {i}"])
            if np.any(channel1 == id) or np.any(channel2 == id):
                return True
                
    return False

class KitchensDatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        dataset_dir = Path(dataset_dir) / "pseudokitchens"

        with (dataset_dir / "info.json").open() as f:
            dataset_info = json.load(f)

        self.ingredients = sorted(dataset_info["object_counts"]["ingredient_counts"].keys())

        def data_getter(directory, number_of_examples):
            def getter(idx):
                instance_id_str = str(idx + 1).zfill(len(str(number_of_examples)))
                image_path = directory / (instance_id_str + ".png")
                image = Image.open(image_path).convert("RGB")

                with (directory / (instance_id_str + ".json")).open() as f:
                    instance_info = json.load(f)
                    if "recipe_idx" in instance_info:
                        class_label = instance_info["recipe_idx"]
                    else:
                        class_label = 0
                with OpenEXR.File(str(directory / (instance_id_str + ".exr"))) as exrfile:
                    manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
                    channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
                    channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

                concept_annotations = []
                for ingredient in self.ingredients:
                    concept_annotation = 0
                    if image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
                        concept_annotation = 1
                    concept_annotations.append(concept_annotation)

                return image, class_label, torch.tensor(concept_annotations, dtype=torch.float32)
            getter.length = number_of_examples
            return getter

        train_img_transform = None
        val_test_img_transform = None
        if foundation_model is None:
            train_img_transform = transforms.resnet_train
            val_test_img_transform = transforms.resnet_val_test

        super().__init__(
            train_getter=data_getter(dataset_dir / "train", dataset_info["train_size"]),
            val_getter=data_getter(dataset_dir / "val", dataset_info["val_size"]),
            test_getter=data_getter(dataset_dir / "test", dataset_info["test_size"]),
            foundation_model=foundation_model,
            train_img_transform=train_img_transform,
            val_test_img_transform=val_test_img_transform,
            representation_cache_dir=dataset_dir,
            model_dir=model_dir,
            device=device
        )

        self.recipeless_train_getter = data_getter(dataset_dir / "recipeless_train", dataset_info["recipeless_train_size"])
        self.recipeless_val_getter = data_getter(dataset_dir / "recipeless_val", dataset_info["recipeless_val_size"])
        self.recipeless_test_getter = data_getter(dataset_dir / "recipeless_test", dataset_info["recipeless_test_size"])
        if self.foundation_model is not None:
            cache_file = Path(dataset_dir) / f"{self.foundation_model_string}_recipeless.pt"
            if cache_file.exists():
                data = torch.load(cache_file)
                self.recipeless_train_x = data["recipeless_train_x"]
                self.recipeless_train_y = data["recipeless_train_y"]
                self.recipeless_train_c = data["recipeless_train_c"]
                self.recipeless_val_x = data["recipeless_val_x"]
                self.recipeless_val_y = data["recipeless_val_y"]
                self.recipeless_val_c = data["recipeless_val_c"]
                self.recipeless_test_x = data["recipeless_test_x"]
                self.recipeless_test_y = data["recipeless_test_y"]
                self.recipeless_test_c = data["recipeless_test_c"]
            else:
                self.recipeless_train_x, self.recipeless_train_y, self.recipeless_train_c = self.run_foundation_model(
                    train_img_transform,
                    model_dir,
                    self.recipeless_train_getter,
                    device)
                self.recipeless_val_x, self.recipeless_val_y, self.recipeless_val_c = self.run_foundation_model(
                    val_test_img_transform,
                    model_dir,
                    self.recipeless_val_getter,
                    device)
                self.recipeless_test_x, self.recipeless_test_y, self.recipeless_test_c = self.run_foundation_model(
                    val_test_img_transform,
                    model_dir,
                    self.recipeless_test_getter,
                    device)
                data = {
                    "recipeless_train_x": self.recipeless_train_x,
                    "recipeless_train_y": self.recipeless_train_y,
                    "recipeless_train_c": self.recipeless_train_c,
                    "recipeless_val_x": self.recipeless_val_x,
                    "recipeless_val_y": self.recipeless_val_y,
                    "recipeless_val_c": self.recipeless_val_c,
                    "recipeless_test_x": self.recipeless_test_x,
                    "recipeless_test_y": self.recipeless_test_y,
                    "recipeless_test_c": self.recipeless_test_c
                }
                torch.save(data, cache_file)

        self.n_concepts = len(self.ingredients)
        self.n_tasks = len(dataset_info["recipes"])
        self.concept_names = self.ingredients

    def recipeless_train_dl(self, additional_concepts=None, use_provided_concepts=True, no_concepts=False):
        if self.foundation_model is not None:
            dataset = self._get_foundation_model_dataset(
                self.recipeless_train_x,
                self.recipeless_train_y,
                self.recipeless_train_c,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        else:
            dataset = CEMDataset(
                self.recipeless_train_getter,
                self.train_img_transform,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        
        return DataLoader(
                dataset,
                batch_size=256,
                num_workers=7)

    def recipeless_val_dl(self, additional_concepts=None, use_provided_concepts=True, no_concepts=False):
        if self.foundation_model is not None:
            dataset = self._get_foundation_model_dataset(
                self.recipeless_val_x,
                self.recipeless_val_y,
                self.recipeless_val_c,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        else:
            dataset = CEMDataset(
                self.recipeless_val_getter,
                self.val_test_img_transform,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        
        return DataLoader(
                dataset,
                batch_size=256,
                num_workers=7)

    def recipeless_test_dl(self, additional_concepts=None, use_provided_concepts=True, no_concepts=False):
        if self.foundation_model is not None:
            dataset = self._get_foundation_model_dataset(
                self.recipeless_test_x,
                self.recipeless_test_y,
                self.recipeless_test_c,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        else:
            dataset = CEMDataset(
                self.recipeless_test_getter,
                self.val_test_img_transform,
                additional_concepts,
                use_provided_concepts,
                no_concepts)
        
        return DataLoader(
                dataset,
                batch_size=256,
                num_workers=7)
