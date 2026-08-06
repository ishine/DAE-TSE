import wesep.models.bsrnn as bsrnn
import wesep.models.dae_bsrnn as dae_bsrnn

_model_registry = {
    "BSRNN": bsrnn.BSRNN,
    "DAE-BSRNN": dae_bsrnn.DAEBSRNN,
}


def get_model(model_name: str):
    if model_name in _model_registry:
        return _model_registry[model_name]
    raise ValueError(f"Unknown model: {model_name}")
