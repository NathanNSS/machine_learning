import os
import fiftyone.zoo as foz
import fiftyone.types as fot

from fiftyone import ViewField as F
from config import DATASET_BASE

#
# OBS: Isso e apenas para aprendizado, pois este script não controla a qualidade do dataset! ele apenas baixa tudo e filtra de acordo com as classes  
#

# Classes desejadas
target_classes = ["person", "car", "dog"]

# Baixa dataset (pode aumentar max_samples para mais imagens)
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    max_samples=500,
    shuffle=True,
    dataset_name="coco_mini_split"
)

# Filtra para conter só as classes desejadas
filtered_view = dataset.filter_labels(
    "ground_truth",
    F("label").is_in(target_classes)
)
filtered_view = filtered_view.match(F("ground_truth.detections").length() > 0)

# Limita para 500 imagens (ou o que quiser)
filtered_view = filtered_view.limit(500)

# Calcula índices para dividir 70/20/10
total = filtered_view.count()
train_count = int(total * 0.7)
val_count = int(total * 0.2)
test_count = total - train_count - val_count

# Cria views
train_view = filtered_view.limit(train_count)
val_view = filtered_view.skip(train_count).limit(val_count)
test_view = filtered_view.skip(train_count + val_count).limit(test_count)

# Diretório base para exportar
export_dir = DATASET_BASE

# Apaga a pasta se existir para evitar conflito (atenção: apaga mesmo!)
if os.path.exists(export_dir):
    import shutil
    shutil.rmtree(export_dir)

# Exporta cada split
for split, view in zip(["train", "val", "test"], [train_view, val_view, test_view]):
    print(f"Exportando {split} com {view.count()} imagens...")
    view.export(
        export_dir=os.path.join(export_dir),
        dataset_type=fot.YOLOv5Dataset,
        label_field="ground_truth",
        split=split,
        classes=target_classes
    )

# Gera arquivo data.yaml
yaml_str = f"""\
path: {export_dir}
train: ./images/train
val: ./images/val
test: ./images/test

nc: {len(target_classes)}
names: {target_classes}
"""

with open(os.path.join(export_dir, "data.yaml"), "w") as f:
    f.write(yaml_str)

dataset_yaml = os.path.join(export_dir, "dataset.yaml")
if os.path.exists(dataset_yaml):
   print("Apagando arquivo dataset.yaml do COCO")
   os.remove(dataset_yaml)

print("Dataset exportado com sucesso!")
print(f"Arquivo data.yaml criado em {export_dir}/data.yaml")
