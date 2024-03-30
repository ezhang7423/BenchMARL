#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm
from .iddpg import Iddpg
from .ippo import Ippo
from .iql import Iql
from .isac import Isac
from .maddpg import Maddpg
from .mappo import Mappo
from .masac import Masac
from .qmix import Qmix
from .vdn import Vdn

classes = [
    "Iddpg",    
    "Ippo",    
    "Iql",    
    "Isac",    
    "Maddpg",    
    "Mappo",    
    "Masac",    
    "Qmix",    
    "Vdn",    
]
