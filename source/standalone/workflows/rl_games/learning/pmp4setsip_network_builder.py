# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class PMP4SetsIPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self._num_pmp4setsip_obs_steps = kwargs.get('numAMPObsSteps', 8)
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(
                        **self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(
                        actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            pmp4setsip_input_shape = kwargs.get('pmp4setsip_input_shape')
            # print(f'pmp4setsip_input_shape: {pmp4setsip_input_shape}')
            # self._build_disc(amp_input_shape)
            # TODO: Implement this
            pmp_upper_body_input_shape = (self._num_pmp4setsip_obs_steps * 40,)
            pmp_lower_body_input_shape = (self._num_pmp4setsip_obs_steps * 59,)
            pmp_right_hand_body_input_shape = (self._num_pmp4setsip_obs_steps * 50,)
            pmp_left_hand_body_input_shape = (self._num_pmp4setsip_obs_steps * 50,)
            pmp_ip_right_hand_body_input_shape = (self._num_pmp4setsip_obs_steps * 75,)
            pmp_ip_left_hand_body_input_shape = (self._num_pmp4setsip_obs_steps * 75,)
            self._build_discs(
                pmp_upper_body_input_shape,
                pmp_lower_body_input_shape,
                pmp_right_hand_body_input_shape,
                pmp_left_hand_body_input_shape,
                pmp_ip_right_hand_body_input_shape,
                pmp_ip_left_hand_body_input_shape
            )

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def eval_discs(
                self,
                pmp_obs_upper,
                pmp_obs_lower,
                pmp_obs_right_hand,
                pmp_obs_left_hand,
                pmp_obs_ip_right_hand,
                pmp_obs_ip_left_hand,
            ):
            disc_mlp_upper_out = self._disc_mlp_upper(pmp_obs_upper)
            disc_logits_upper = self._disc_logits_upper(disc_mlp_upper_out)

            disc_mlp_lower_out = self._disc_mlp_lower(pmp_obs_lower)
            disc_logits_lower = self._disc_logits_lower(disc_mlp_lower_out)

            disc_mlp_right_hand_out = self._disc_mlp_right_hand(pmp_obs_right_hand)
            disc_logits_right_hand = self._disc_logits_right_hand(disc_mlp_right_hand_out)

            disc_mlp_left_hand_out = self._disc_mlp_left_hand(pmp_obs_left_hand)
            disc_logits_left_hand = self._disc_logits_left_hand(disc_mlp_left_hand_out)

            disc_mlp_ip_right_hand_out = self._disc_mlp_ip_right_hand(pmp_obs_ip_right_hand)
            disc_logits_ip_right_hand = self._disc_logits_ip_right_hand(disc_mlp_ip_right_hand_out)

            disc_mlp_ip_left_hand_out = self._disc_mlp_ip_left_hand(pmp_obs_ip_left_hand)
            disc_logits_ip_left_hand = self._disc_logits_ip_left_hand(disc_mlp_ip_left_hand_out)

            return (
                disc_logits_upper,
                disc_logits_lower,
                disc_logits_right_hand,
                disc_logits_left_hand,
                disc_logits_ip_right_hand,
                disc_logits_ip_left_hand
            )

        def get_discs_logit_weights(self):
            return (
                torch.flatten(self._disc_logits_upper.weight),
                torch.flatten(self._disc_logits_lower.weight),
                torch.flatten(self._disc_logits_right_hand.weight),
                torch.flatten(self._disc_logits_left_hand.weight),
                torch.flatten(self._disc_logits_ip_right_hand.weight),
                torch.flatten(self._disc_logits_ip_left_hand.weight)
            )

        def get_discs_weights(self):
            weights_upper = []
            for m in self._disc_mlp_upper.modules():
                if isinstance(m, nn.Linear):
                    weights_upper.append(torch.flatten(m.weight))

            weights_upper.append(torch.flatten(self._disc_logits_upper.weight))

            weights_lower = []
            for m in self._disc_mlp_lower.modules():
                if isinstance(m, nn.Linear):
                    weights_lower.append(torch.flatten(m.weight))

            weights_lower.append(torch.flatten(self._disc_logits_lower.weight))

            weights_right_hand = []
            for m in self._disc_mlp_right_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_right_hand.append(torch.flatten(m.weight))

            weights_right_hand.append(torch.flatten(self._disc_logits_right_hand.weight))

            weights_left_hand = []
            for m in self._disc_mlp_left_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_left_hand.append(torch.flatten(m.weight))

            weights_left_hand.append(torch.flatten(self._disc_logits_left_hand.weight))

            weights_ip_right_hand = []
            for m in self._disc_mlp_ip_right_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_ip_right_hand.append(torch.flatten(m.weight))

            weights_ip_right_hand.append(torch.flatten(self._disc_logits_ip_right_hand.weight))

            weights_ip_left_hand = []
            for m in self._disc_mlp_ip_left_hand.modules():
                if isinstance(m, nn.Linear):
                    weights_ip_left_hand.append(torch.flatten(m.weight))

            weights_ip_left_hand.append(torch.flatten(self._disc_logits_ip_left_hand.weight))

            return (
                weights_upper,
                weights_lower,
                weights_right_hand,
                weights_left_hand,
                weights_ip_right_hand,
                weights_ip_left_hand
            )

        def _build_discs(
                self,
                input_shape_upper_body,
                input_shape_lower_body,
                input_shape_right_hand_body,
                input_shape_left_hand_body,
                input_shape_ip_right_hand_body,
                input_shape_ip_left_hand_body
            ):
            self._disc_mlp_upper = nn.Sequential()
            self._disc_mlp_lower = nn.Sequential()
            self._disc_mlp_right_hand = nn.Sequential()
            self._disc_mlp_left_hand = nn.Sequential()
            self._disc_mlp_ip_right_hand = nn.Sequential()
            self._disc_mlp_ip_left_hand = nn.Sequential()

            mlp_upper_args = {
                'input_size': input_shape_upper_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_lower_args = {
                'input_size': input_shape_lower_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_right_hand_args = {
                'input_size': input_shape_right_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_left_hand_args = {
                'input_size': input_shape_left_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_ip_right_hand_args = {
                'input_size': input_shape_ip_right_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            mlp_ip_left_hand_args = {
                'input_size': input_shape_ip_left_hand_body[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }

            self._disc_mlp_upper = self._build_mlp(**mlp_upper_args)
            self._disc_mlp_lower = self._build_mlp(**mlp_lower_args)
            self._disc_mlp_right_hand = self._build_mlp(**mlp_right_hand_args)
            self._disc_mlp_left_hand = self._build_mlp(**mlp_left_hand_args)
            self._disc_mlp_ip_right_hand = self._build_mlp(**mlp_ip_right_hand_args)
            self._disc_mlp_ip_left_hand = self._build_mlp(**mlp_ip_left_hand_args)

            mlp_out_size = self._disc_units[-1]
            self._disc_logits_upper = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_lower = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_right_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_left_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_ip_right_hand = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits_ip_left_hand = torch.nn.Linear(mlp_out_size, 1)

            mlp_init_upper = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_upper.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_upper(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            mlp_init_lower = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_lower.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_lower(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            mlp_init_right_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_right_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_right_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_left_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_left_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_left_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_ip_right_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_ip_right_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_ip_right_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            mlp_init_ip_left_hand = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_ip_left_hand.modules():
                if isinstance(m, nn.Linear):
                    mlp_init_ip_left_hand(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.nn.init.uniform_(
                self._disc_logits_upper.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_upper.bias)

            torch.nn.init.uniform_(
                self._disc_logits_lower.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_lower.bias)

            torch.nn.init.uniform_(
                self._disc_logits_right_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_right_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_left_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_left_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_ip_right_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_ip_right_hand.bias)

            torch.nn.init.uniform_(
                self._disc_logits_ip_left_hand.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_ip_left_hand.bias)

            return

    def build(self, name, **kwargs):
        net = PMP4SetsIPBuilder.Network(self.params, **kwargs)
        return net
