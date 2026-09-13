import unittest
import torch
from audioseal_st.models import shape_residual


class ProjectionTest(unittest.TestCase):
    def test_per_clip_energy_budget_and_gradient(self):
        torch.manual_seed(9)
        x=torch.randn(2,1,4096)*.05
        r=torch.randn_like(x,requires_grad=True)
        y=shape_residual(x,r,22,.2)
        snr=10*torch.log10(x.square().mean(-1)/y.square().mean(-1))
        self.assertTrue(bool((snr>=21.999).all()))
        y.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(r.grad).all()))

    def test_silence_stays_silent(self):
        x=torch.zeros(1,1,4096)
        r=torch.randn_like(x,requires_grad=True)
        y=shape_residual(x,r)
        self.assertEqual(float(y.abs().max()),0)
        y.sum().backward()
        self.assertTrue(bool(torch.isfinite(r.grad).all()))


if __name__=='__main__':unittest.main()
