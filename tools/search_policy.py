import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from argparse import ArgumentParser, Namespace

import torch.cuda
import wandb

from runtime.agent.agent import GenericAgentConfig
from runtime.agent.pq_agent import PruningQuantizationAgent, PruningQuantizationAgentConfig
from runtime.agent.pruning_agent import IterativeSingleLayerPruningAgent, IndependentSingleLayerPruningAgent
from runtime.agent.quantization_agent import IterativeSingleLayerQuantizationAgent, QuantizationAgentConfig, \
    IndependentSingleLayerQuantizationAgent, Q3AIndependent
from runtime.torch_recipe import TorchConfiguration, TorchRecipe
from tools.util.model_provider import load_model
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.utils.torch_utils import initialize_weights

ENV_JOB_ID = "SLURM_JOB_ID"


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        help="Model for repo model use 'model@repo'",
        type=str,
        default="resnet18_cifar"
    )

    parser.add_argument(
        "--ckpt_load_path",
        dest="ckpt_load_path",
        help="Path to the model checkpoint data to load from. (To continue training from or test)",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Path to the directory containing the data set.",
        type=str,
        default="/home/yichao/Desktop/tick/model_compression/torch_prune_example/data.yaml", #'./data/'
    )

    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        help="Directory to save logs and results to",
        type=str,
        default="./logs"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None
    )

    parser.add_argument(
        "--add_search_identifier",
        type=str,
        default=""
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="dummy"
    )

    parser.add_argument(
        "--alg_config",
        metavar="KEY=VALUE",
        nargs="+",
        help="Specify algorithmic configuration parameters and hyperparameters in key=value format. "
             "Available parameters depend on selected agent."
    )

    return parser.parse_args()


def agent_provider(agent_identifier: str, device: torch.device, config: dict):
    if agent_identifier == "iterative-single-layer-pruning":
        return IterativeSingleLayerPruningAgent(device, agent_config=GenericAgentConfig(**config))
    if agent_identifier == "independent-single-layer-pruning":
        return IndependentSingleLayerPruningAgent(device, agent_config=GenericAgentConfig(**config))
    if agent_identifier == "iterative-single-layer-quantization":
        return IterativeSingleLayerQuantizationAgent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "independent-single-layer-quantization":
        return IndependentSingleLayerQuantizationAgent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "independent-3-action-quantization":
        return Q3AIndependent(device, agent_config=QuantizationAgentConfig(**config))
    if agent_identifier == "pruning-quantization-agent":
        return PruningQuantizationAgent(device, agent_config=PruningQuantizationAgentConfig(**config))
    raise Exception("Not a valid agent_identifier")


def fetch_env_config():
    env_config = dict()
    if ENV_JOB_ID in os.environ:
        env_config["JOB_ID"] = os.getenv(ENV_JOB_ID)
    return env_config


###Yolo model adaption
def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)
if __name__ == '__main__':
    args = parse_arguments()
    alg_config = dict(map(lambda s: s.split('='), args.alg_config))
    #device = torch.device('cpu') #None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    # from ultralytics import YOLO
    # model = YOLO('/home/yichao/Desktop/tick/model_compression/torch_prune_example/original.pt')
    # metrics=model.val(data="/home/yichao/Desktop/tick/model_compression/torch_prune_example/data.yaml")

    agent = agent_provider(args.agent, device, alg_config)
    search_id = f"{agent.get_search_id()}_{args.add_search_identifier}"

    runtime_config = TorchConfiguration(target_device=device,
                                        log_dir=args.log_dir,
                                        data_dir=args.data_dir,
                                        enabled_methods=agent.supported_method(),
                                        search_identifier=search_id,
                                        **agent.config_overrides(),
                                        **alg_config)

    run_config = vars(runtime_config) | agent.config() | fetch_env_config()
    #wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=run_config, name=search_id)

    model, _ = load_model(args.model, num_classes=runtime_config.num_classes, checkpoint_path=args.ckpt_load_path)
    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)
    model.overrides['data']='/home/yichao/Desktop/tick/model_compression/torch_prune_example/data.yaml'
    runtime_controller, original_model_handle = TorchRecipe(agent, runtime_config).construct_application(model)
    runtime_controller.search(args.episodes, original_model_handle)
