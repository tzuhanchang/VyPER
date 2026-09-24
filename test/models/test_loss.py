import torch


def test_DiffusionLoss_num_graphs():
    from VyPER.models.loss import DiffusionLoss

    # 4 graphs: graphs 1 and 3 have no neutrinos
    nu_loss = torch.tensor([1., 2., 3., 4.])
    nu_batch = torch.tensor([0, 0, 2, 2])

    out = DiffusionLoss(nu_loss, nu_batch, reduction='sum', num_graphs=4)
    assert torch.equal(out, torch.tensor([3., 0., 7., 0.]))
