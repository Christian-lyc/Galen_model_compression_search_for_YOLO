import ast
import os
from argparse import Namespace, ArgumentParser
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import torch
import wandb
import ast
import json
from runtime.data.data_provider import CIFAR10Provider, YOLODetectionProvider
from runtime.data.imagenet_provider import ImageNetDataProvider
from tools.util import model_provider
from tools.util.compress import compress_model
from tools.util.model_provider import load_checkpoint
from tools.util.trainer import Trainer
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.utils.torch_utils import initialize_weights
import torch.nn as nn
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
        "--compressed_ckpt",
        dest="compressed_ckpt",
        help="Path to the model checkpoint data to load from. (To continue training from or test)",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--policy",
        help="Specify a compression policy / episode logging dict to compress the model with",
        default=None,
        required=False
    )

    parser.add_argument(
        "--store_dir",
        type=str,
        default="./results/checkpoints/cifar/"
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Path to the directory containing the data set.",
        type=str,
        default="/home/yichao/Desktop/tick/model_compression/torch_prune_example/data.yaml",
    )

    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        type=str,
        default="./logs/",
    )

    parser.add_argument(
        "--log_name",
        dest="log_name",
        type=str,
        default="train.out",
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
        "--add_train_identifier",
        type=str,
        default=""
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="tick"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=6
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num_worker",
        type=int,
        default=4
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--normalization_constants",
        type=str,
        default="([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])"
    )

    parser.add_argument(
        "--train_lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--train_mom",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4
    )

    parser.add_argument(
        "--use_adadelta",
        action="store_true"
    )


    return parser.parse_args()


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
    default_frozen_layers = {
        'p-lin': ['fc'],
        'q-mixed': [
            'fc', 'model.22.cv2.0.0.conv', 'model.22.cv2.0.1.conv', 'model.22.cv2.0.2',
            'model.22.cv2.1.0.conv', 'model.22.cv2.1.1.conv', 'model.22.cv2.1.2',
            'model.22.cv2.2.0.conv', 'model.22.cv2.2.1.conv', 'model.22.cv2.2.2',
            'model.22.cv3.0.0.conv', 'model.22.cv3.0.1.conv', 'model.22.cv3.0.2',
            'model.22.cv3.1.0.conv', 'model.22.cv3.1.1.conv', 'model.22.cv3.1.2',
            'model.22.cv3.2.0.conv', 'model.22.cv3.2.1.conv', 'model.22.cv3.2.2',
            'model.22.dfl.conv'
            ],
        'p-conv': [
            'model.22.cv2.0.0.conv', 'model.22.cv2.0.1.conv', 'model.22.cv2.0.2',
            'model.22.cv2.1.0.conv', 'model.22.cv2.1.1.conv', 'model.22.cv2.1.2',
            'model.22.cv2.2.0.conv', 'model.22.cv2.2.1.conv', 'model.22.cv2.2.2',
            'model.22.cv3.0.0.conv', 'model.22.cv3.0.1.conv', 'model.22.cv3.0.2',
            'model.22.cv3.1.0.conv', 'model.22.cv3.1.1.conv', 'model.22.cv3.1.2',
            'model.22.cv3.2.0.conv', 'model.22.cv3.2.1.conv', 'model.22.cv3.2.2',
            'model.22.dfl.conv'
            ],
        'q-int8': [
            'model.22.cv2.0.0.conv', 'model.22.cv2.0.1.conv', 'model.22.cv2.0.2',
            'model.22.cv2.1.0.conv', 'model.22.cv2.1.1.conv', 'model.22.cv2.1.2',
            'model.22.cv2.2.0.conv', 'model.22.cv2.2.1.conv', 'model.22.cv2.2.2',
            'model.22.cv3.0.0.conv', 'model.22.cv3.0.1.conv', 'model.22.cv3.0.2',
            'model.22.cv3.1.0.conv', 'model.22.cv3.1.1.conv', 'model.22.cv3.1.2',
            'model.22.cv3.2.0.conv', 'model.22.cv3.2.1.conv', 'model.22.cv3.2.2',
            'model.22.dfl.conv'
            ],
        }


# Convert to string for kwargs
    default_frozen_layers_str = json.dumps(default_frozen_layers)

# Now you can use it like this:
    frozen_layers = ast.literal_eval(
     default_frozen_layers_str
    )
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model, model_name = model_provider.load_model(args.model, args.num_classes, args.ckpt_load_path)

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)
    model.overrides['data']='/home/yichao/Desktop/tick/model_compression/torch_prune_example/data.yaml'
    model.model.to(device)
    data_provider = None
    if args.dataset == "cifar":
        data_provider = CIFAR10Provider(device,
                                        seed=args.seed,
                                        batch_size=args.batch_size,
                                        data_dir=args.data_dir,
                                        num_workers=args.num_worker,
                                        split_ratio=args.split_ratio,
                                        image_normalization_constants=ast.literal_eval(args.normalization_constants))

    if args.dataset == "imagenet":
        data_provider = ImageNetDataProvider(device,
                                             data_dir=args.data_dir,
                                             batch_size=args.batch_size,
                                             seed=args.seed,
                                             num_workers=args.num_worker)

    if args.dataset == "tick":
        data_provider=YOLODetectionProvider(device,
                                             data_dir=args.data_dir,
                                             batch_size=args.batch_size,
                                             seed=args.seed,
                                             num_workers=args.num_worker,
                                             model=model)



    #print(model)

    epochs = args.epochs
    

    trainer = Trainer(
        data_provider,
        device,
        args,
        train_epochs=epochs,
        train_lr=args.train_lr,
        train_mom=args.train_mom,
        class_num=args.num_classes,
        use_adadelta=args.use_adadelta,
        store_dir=args.store_dir,
        add_identifier=args.add_train_identifier,
        model_name=model_name,
        log_dir=args.log_dir,
        log_file_name=args.log_name,
        model=model
    )

    #wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args),
    #           name=trainer.create_identifier(args.epochs))

    protocol = None
    if args.policy:

        enabled_methods = tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])
        model.model, protocol = compress_model(model.model, args.policy, data_provider, device, enabled_methods,frozen_layers)
        checkpoint_path = args.compressed_ckpt
        if checkpoint_path is not None:
            model.load_state_dict(load_checkpoint(checkpoint_path))
        #print(model)

    best_model = trainer.train(model=model.model)

    test_acc = trainer.test(best_model)
    trainer.store_logs({
        "compression_protocol": protocol
    })
    print(f"Acc. for test: {test_acc}")
