import torch
from Patchcore_wrapper import PatchCoreExport, pc, mean, std
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.devtools.backend_debug import get_delegation_info


def lower_and_save(module, example_inputs, path):
    with torch.no_grad():
        ep = torch.export.export(module, example_inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    et = edge.to_executorch()
    with open(path, "wb") as f:
        f.write(et.buffer)
    return edge

    
if __name__ == "__main__":
    module = PatchCoreExport(pc, mean, std).eval()

    x = torch.rand(1, 3, 256, 256)

    edge = lower_and_save(module, (x,), path="metal_plate_double_fp32.pte")

    print(get_delegation_info(edge.exported_program().graph_module))
