from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import MVTecAD


model = Patchcore(
    backbone="resnet18",
    layers=("layer2","layer3"),
    pre_trained=True,
    num_neighbors=9,
    coreset_sampling_ratio=0.01

)
datamodule = MVTecAD(root="/home/prajesh_gogoi/my_projects/torch_projects/arm_challenge/Phase4_Pipeline/dataset/metal_plate/", category="metal_plate")

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

results = engine.test(model=model, datamodule=datamodule)
print(results)

