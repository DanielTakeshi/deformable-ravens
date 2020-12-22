from ravens.agents.dummy import DummyAgent
from ravens.agents.transporter import (
        OriginalTransporterAgent,
        GoalTransporterAgent,
        GoalNaiveTransporterAgent,
        GoalSuperNaiveTransporterAgent,
        NoTransportTransporterAgent,
        PerPixelLossTransporterAgent,
)
from ravens.agents.regression import PickThenPlaceRegressionAgent
from ravens.agents.conv_mlp import PickPlaceConvMlpAgent
from ravens.agents.gt_state import GtStateAgent
from ravens.agents.gt_state_2_step import GtState2StepAgent

# Daniel: adding a bunch of transporter-goal agents.
names = {'dummy':                   DummyAgent,
         'transporter':             OriginalTransporterAgent,
         'transporter-goal':        GoalTransporterAgent,
         'transporter-goal-naive':  GoalNaiveTransporterAgent,
         'transporter-goal-snaive': GoalSuperNaiveTransporterAgent,
         'no_transport':            NoTransportTransporterAgent,
         'per_pixel_loss':          PerPixelLossTransporterAgent,
         'regression':              PickThenPlaceRegressionAgent,
         'conv_mlp':                PickPlaceConvMlpAgent,
         'gt_state':                GtStateAgent,
         'gt_state_2_step':         GtState2StepAgent,}
