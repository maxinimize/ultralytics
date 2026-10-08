"""Check frozen FGD, real YOLO detection gradients and existing ART PGD/BIM/MIM on one batch."""

import argparse

import torch

from ultralytics import YOLO
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.cfg import get_cfg
from ultralytics.defenses.hgd_fgd import FGDDetectorWrapper, apply_fgd, load_fgd_denoiser


def main():
    """Run actual models/attacks without mocking devices or writing training checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fgd_weights", required=True)
    parser.add_argument("--model", default="yolo12l.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    fgd, _ = load_fgd_denoiser(args.fgd_weights, device)
    detector = YOLO(args.model).model.to(device).float()
    detector.args = get_cfg(overrides=detector.args)
    detector.requires_grad_(True)
    for name, parameter in detector.named_parameters():
        if ".dfl" in name:
            parameter.requires_grad_(False)
    wrapper = FGDDetectorWrapper(fgd, detector)
    before = {key: value.clone() for key, value in fgd.state_dict().items()}
    images = torch.rand(2, 3, args.imgsz, args.imgsz, device=device)
    batch = dict(
        img=images,
        batch_idx=torch.tensor([0, 1], device=device),
        cls=torch.zeros(2, 1, device=device),
        bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.4]] * 2, device=device),
    )
    detector.train()
    x = images.clone().requires_grad_(True)
    loss, _ = wrapper.loss({**batch, "img": x})
    gradient = torch.autograd.grad(loss.sum(), x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    print(f"input gradient through FGD: PASS (L1={gradient.abs().sum().item():.6g})")

    forwards = []
    handle = fgd.register_forward_hook(lambda module, inputs, output: forwards.append(output.requires_grad))
    for name in ("pgd", "bim", "mim"):
        attacker = build_attacker(name, model=wrapper, img_size=args.imgsz, epoch=1)
        detector.train()  # The real trainer switches the detector independently of the attack wrapper.
        states = [p.requires_grad for p in detector.parameters()]
        detector.requires_grad_(False)
        forwards.clear()
        adv = run_attack_on_batch(attacker, batch)
        for p, state in zip(detector.parameters(), states):
            p.requires_grad_(state)
        assert adv is not None and torch.isfinite(adv).all()
        assert forwards and all(forwards)
        assert (adv - images).abs().max() <= attacker.epsilon + 1e-6
        assert detector.training and wrapper.training and not fgd.training
        print(f"{name.upper()} attacks FGD + YOLO: PASS ({len(forwards)} differentiable FGD forwards)")
    handle.remove()

    optimizer = torch.optim.SGD([p for p in detector.parameters() if p.requires_grad], lr=1e-3)
    fgd_ids = {id(p) for p in fgd.parameters()}
    assert not any(id(p) in fgd_ids for group in optimizer.param_groups for p in group["params"])
    weight = next(p for p in detector.parameters() if p.requires_grad)
    yolo_before = weight.detach().clone()
    loss, _ = detector.loss({**batch, "img": apply_fgd(fgd, adv)})
    optimizer.zero_grad()
    loss.sum().backward()
    assert all(p.grad is None for p in fgd.parameters())
    optimizer.step()
    difference = (weight.detach() - yolo_before).abs().max().item()
    assert difference > 0
    assert all(torch.equal(value, before[key]) for key, value in fgd.state_dict().items())
    assert not any(p.requires_grad for p in fgd.parameters())
    assert detector.training and not fgd.training
    print("FGD parameter frozen: PASS (max_parameter_difference=0)")
    print("FGD grad None: PASS")
    print("FGD optimizer exclusion: PASS")
    print(f"YOLO trainable: PASS (max_parameter_difference={difference:.6g})")
    print("FGD eval mode / YOLO train mode: PASS")


if __name__ == "__main__":
    main()
