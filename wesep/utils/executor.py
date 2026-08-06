# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def process_additional(batch: dict, device: torch.device) -> dict:
    additional = batch.get("additional", {})
    if additional == {}:
        return {}

    assert 'data' in additional
    for key in additional['data'].keys():
        if isinstance(additional['data'][key], torch.Tensor):
            additional['data'][key] = additional['data'][key].to(device)

    if 'mix_enroll' in additional['data'].keys():
        additional['data']['mix_enroll'] = [
            _.to(device) for _ in additional['data']['mix_enroll']
        ]  # List[torch.Tensor]

    return additional
