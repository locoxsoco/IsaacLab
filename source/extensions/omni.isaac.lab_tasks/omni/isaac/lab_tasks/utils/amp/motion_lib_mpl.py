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

import numpy as np

from omni.isaac.lab_tasks.utils.amp.poselib.rotation3d import *

from omni.isaac.lab_tasks.utils.torch_jit_utils import quat_to_exp_map, quat_to_angle_axis, normalize_angle

from omni.isaac.lab_tasks.utils.amp.motion_lib import MotionLib

DOF_BODY_IDS_MPL = [1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                    51, 52, 53, 54]
DOF_OFFSETS_MPL = [0,  3,  6,  9, 10,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 31, 32, 35, 36,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                   51, 52, 53, 54, 55, 56, 57, 58, 61, 62,
                   65, 68, 69, 72]

class MotionLibMPL(MotionLib):    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = DOF_BODY_IDS_MPL
        dof_offsets = DOF_OFFSETS_MPL
        dof_x_axis_ids = [11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48]
        dof_y_axis_ids = [4, 6, 27, 29, 50, 53]
        dof_z_axis_ids = [7, 8, 9, 10, 18, 22, 30, 31, 32, 33, 41, 45]

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                if(body_id in dof_x_axis_ids):
                    joint_theta = joint_theta * joint_axis[..., 0] # joint is along x axis
                if(body_id in dof_y_axis_ids):
                    joint_theta = joint_theta * joint_axis[..., 1] # joint is along y axis
                if(body_id in dof_z_axis_ids):
                    joint_theta = joint_theta * joint_axis[..., 2] # joint is along z axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = DOF_BODY_IDS_MPL
        dof_offsets = DOF_OFFSETS_MPL
        dof_x_axis_ids = [11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48]
        dof_y_axis_ids = [4, 6, 27, 29, 50, 53]
        dof_z_axis_ids = [7, 8, 9, 10, 18, 22, 30, 31, 32, 33, 41, 45]

        dof_vel = np.zeros([self._num_dof])

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel.numpy()

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                if(body_id in dof_x_axis_ids):
                    dof_vel[joint_offset] = joint_vel[0] # joint is along x axis
                if(body_id in dof_y_axis_ids):
                    dof_vel[joint_offset] = joint_vel[1] # joint is along y axis
                if(body_id in dof_z_axis_ids):
                    dof_vel[joint_offset] = joint_vel[2] # joint is along z axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel