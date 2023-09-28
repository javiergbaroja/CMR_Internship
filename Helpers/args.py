import argparse
import sys

def get_base_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_folder', type=str, default="Results")
    parser.add_argument('--log_folder', type=str, default="logs_folder")
    parser.add_argument('--data_folder', type=str, default="Data")

    return parser

def get_args_training():

    # General Arguments
    parser = get_base_parser()

    # Data Arguments
    parser.add_argument('--use_predifined_sets', action='store_true')
    parser.add_argument('--fixed_sets', type=str, default=None)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--augment', action='store_true')

    # Model arguments
    parser.add_argument('--views', type=str, default="3D")
    parser.add_argument('--network', type=str)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--pretrained', type=str, default=None)    
    parser.add_argument('--return_logits', action='store_true')
    parser.add_argument('--merge_bilstm', type=str, default="concat")
    parser.add_argument('--n_lstm_layers', type=int, default=1)

    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=-1)


    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default="adam")

    # Training arguments
    parser.add_argument('--n_cross_val', type=int, default=None)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--label_method', type=str, default="SD")
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--n_channels', type=int, default=-1)
    parser.add_argument("--n_slices_central", type=int, default=-1)
    parser.add_argument("--n_slices_window", type=int, default=-1)
    parser.add_argument("--all_test_slices", action='store_true')
    
    args = parser.parse_args()

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        print('Not Running in Debug Mode')
    elif gettrace():
        print('Running in Debug Mode')
        args.log_folder = "del"
    else:
        ValueError("Error checking if running in Debug Mode")

    return args

def get_args_volume_save():

    # General Arguments
    parser = get_base_parser()

    # Dir Arguments
    parser.add_argument("--results_folder", type=str)
    parser.add_argument("--factor", type=float, default=-1) # Should be 3. for unet, 6. for BiLSTM and 9. for multi_BiLSTM

    args = parser.parse_args()

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        print('Not Running in Debug Mode')
    elif gettrace():
        print('Running in Debug Mode')
        args.log_folder = "del"
    else:
        ValueError("Error checking if running in Debug Mode")

    return args
    