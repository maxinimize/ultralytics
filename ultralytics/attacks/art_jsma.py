import torch

from ultralytics.attacks.attacker import Attacker
from .attack_bridge import build_classifier_estimator, normalize_classifier_targets

from art.attacks.evasion import SaliencyMapMethod


class ARTJSMA(Attacker):
    """
    JSMA through a YOLO single-label classifier proxy.
    """
    attack_mode = "classifier"
    requires_targets = True
    target_format = "single_label"

    def __init__(
        self,
        model,
        config=None,
        target=None,
        theta=0.1,
        gamma=1.0,
        epsilon=0.1,
        batch_size=1,
        img_size=640,
        verbose=False,
    ):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.proxy_model, self.estimator = build_classifier_estimator(self.model, img_size=img_size)
        effective_theta = epsilon if epsilon is not None else theta

        import inspect
        jsma_kwargs = {
            "classifier": self.estimator,
            "theta": effective_theta,
            "gamma": gamma,
            "batch_size": batch_size,
        }
        sig = inspect.signature(SaliencyMapMethod.__init__)
        if "verbose" in sig.parameters:
            jsma_kwargs["verbose"] = verbose

        self.attack = SaliencyMapMethod(**jsma_kwargs)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        labels = normalize_classifier_targets(targets)
        x_adv_np = self.attack.generate(x=x.detach().cpu().numpy(), y=labels)
        return torch.from_numpy(x_adv_np).to(self.device).type_as(x)
