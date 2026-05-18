import torch

from ultralytics.attacks.attacker import Attacker
from .attack_bridge import build_classifier_estimator, normalize_classifier_targets

from art.attacks.evasion import DeepFool


class ARTDeepFool(Attacker):
    """
    DeepFool through a YOLO single-label classifier proxy.
    """
    attack_mode = "classifier"
    requires_targets = True
    target_format = "single_label"

    def __init__(self, model, config=None, target=None, epsilon=0.05, img_size=640):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.proxy_model, self.estimator = build_classifier_estimator(self.model, img_size=img_size)

        try:
            self.attack = DeepFool(estimator=self.estimator, epsilon=epsilon)
        except TypeError:
            self.attack = DeepFool(classifier=self.estimator, epsilon=epsilon)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        labels = normalize_classifier_targets(targets)
        x_adv_np = self.attack.generate(x=x.detach().cpu().numpy(), y=labels)
        return torch.from_numpy(x_adv_np).to(self.device).type_as(x)
