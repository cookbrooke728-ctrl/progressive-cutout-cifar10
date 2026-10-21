"""Choose an execution backend before formal training, never from test accuracy."""
import copy
import gc
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from cutout_lab.data import prepare
from cutout_lab.model import ResNet20
from cutout_lab.utils import dump_json, seed_everything

source, _, split, stats = prepare()
ids = split["probe"][:128]
x = torch.tensor(source.data[ids].transpose(0, 3, 1, 2).copy(), device="cuda").float() / 255.
x = ((x - torch.tensor(stats["mean"], device="cuda")[None, :, None, None]) /
     torch.tensor(stats["std"], device="cuda")[None, :, None, None]).contiguous(memory_format=torch.channels_last)
y = torch.tensor([source.targets[i] for i in ids], device="cuda")
results, reference = {}, None
for backend in ("eager", "fused", "cudagraphs"):
    seed_everything(42)
    model = ResNet20().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, weight_decay=.0005,
                                fused=(backend != "eager"))
    compiled = torch.compile(model, backend="cudagraphs", fullgraph=True) if backend == "cudagraphs" else model
    try:
        for i in range(60):
            if i == 10:
                torch.cuda.synchronize()
                beginning = time.perf_counter()
            if backend == "cudagraphs":
                torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            output = compiled(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - beginning
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if reference is None:
            reference = state
        difference = max((v.float() - reference[k].float()).abs().max().item() for k, v in state.items())
        results[backend] = {"seconds_per_step": seconds / 50, "final_loss": loss.item(), "max_abs_state_diff_from_eager": difference}
    except Exception as error:
        results[backend] = {"error": str(error)}
    print(backend, results[backend], flush=True)
    del model, optimizer, compiled
    gc.collect()
    torch.cuda.empty_cache()
dump_json("validation/backend_benchmark.json", results)
