import pytest
import torch


@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory):
    """A randomly initialised model saved as a checkpoint (validity must not depend on training)."""
    from wtds.model import QNet
    torch.manual_seed(0)
    m = QNet()
    p = tmp_path_factory.mktemp("ck") / "random.pt"
    torch.save({"online": m.state_dict(), "model_cfg": m.config, "env_kwargs": {}}, p)
    return str(p)
