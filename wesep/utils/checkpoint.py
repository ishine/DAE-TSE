import torch


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator"):
    assert type in ["generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
