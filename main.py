import os
import numpy as np
import pandas as pd
import argparse
import logging
from running import Rep_Learning, Supervised

import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, print_title
from Models.EEG_Analysis import generate_important_timepoints_data, get_dataset_fs

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
# 6.25修改
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# --------------------------------------------------- I/O --------------------------------------------------------------
parser.add_argument('--data_dir', default='Dataset/Crowdsource', choices={'Dataset/Crowdsource', 'Dataset/DREAMER',
                                                                          'Dataset/STEW', 'Dataset/TUAB'}, help='Data directory')
# add new parser for TUAB and TUEV root
parser.add_argument('--sub_dir', default='edf/processed', help='Subdirectory under data_dir (e.g., edf/processed for TUAB/TUEV)')

parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Time-stamped directories will be created inside.')
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Parameters and Hyperparameter ----------------------------------------------
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy'}, default='loss', help='Metric used for best epoch')
# -------------------------------------------------- EEG-JEPA ----------------------------------------------------------
parser.add_argument('--Training_mode', default='Rep-Learning', choices={'Rep-Learning', 'Initialization', 'Supervised'})
parser.add_argument('--Pre_Training', default='In-domain', choices={'In-domain', 'Cross-domain'})
parser.add_argument('--Input_Embedding', default=['C'], choices={'T', 'C', 'C-T'}, help="Input Embedding Architecture")
parser.add_argument('--Pos_Embedding', default=['Sin'], choices={'Sin', 'Emb'}, help="Position Embedding Architecture")

parser.add_argument('--Encoder', default=['T'], choices={'T', 'C', 'C-T'}, help="Context/Target Encoder Architecture")
parser.add_argument('--layers', type=int, default=4, help="Number of layers for the context/target encoders")

parser.add_argument('--pre_layers', type=int, default=2, help="Number of layers for the Predictor")
parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")
parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")

parser.add_argument('--patch_size', type=int, default=8, help='size')
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of feedforward network of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# add a new parser for the timepoints load path
parser.add_argument('--precomputed_timepoints_path', type=str, default=None, help='Path to the precomputed important timepoints file (.pkl)')

# 6.25修改，新增参数
parser.add_argument('--thresholds', nargs='+', type=float, default=[0.75, 0.50, 0.25, 0.05],
                    help='List of thresholds to test')
parser.add_argument('--no_consistency', action='store_true',
                    help='Ablate A(t): set cross-channel consistency to 1 (S=I)')

# 获取输入的参数
args = parser.parse_args()
# 初始化变量，包括两个字符串
All_Results = ['Datasets', 'FC_layer']

# for tuab dataset
def load_tuab_with_memmap(config):
    """Load TUAB data using memory mapping to avoid OOM."""
    import numpy as np
    import os
    Data = {}
    data_path = config['data_dir'] + '/' + config['problem']
    train_data_path = os.path.join(data_path, 'train_data.npy')
    if not os.path.exists(train_data_path):
        logger.error(f"TUAB numpy files not found at {data_path}")
        logger.error("Please run Dataset/tuab_preprocess.py first to generate numpy files!")
        raise FileNotFoundError(f"TUAB numpy files not found. Run preprocessing first.")
    logger.info("Loading TUAB data with memory mapping...")
    Data['train_data'] = np.load(os.path.join(data_path, 'train_data.npy'), mmap_mode='r', allow_pickle=True)
    Data['train_label'] = np.load(os.path.join(data_path, 'train_label.npy'), allow_pickle=True)
    Data['val_data'] = np.load(os.path.join(data_path, 'val_data.npy'), mmap_mode='r', allow_pickle=True)
    Data['val_label'] = np.load(os.path.join(data_path, 'val_label.npy'), allow_pickle=True)
    Data['All_train_data'] = np.load(os.path.join(data_path, 'All_train_data.npy'), mmap_mode='r', allow_pickle=True)
    Data['All_train_label'] = np.load(os.path.join(data_path, 'All_train_label.npy'), allow_pickle=True)
    Data['test_data'] = np.load(os.path.join(data_path, 'test_data.npy'), mmap_mode='r', allow_pickle=True)
    Data['test_label'] = np.load(os.path.join(data_path, 'test_label.npy'), allow_pickle=True)
    Data['max_len'] = Data['train_data'].shape[2]

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))
    return Data


if __name__ == '__main__':
    THRESHOLDS = args.thresholds
    config = Setup(args)  # configuration dictionary
    # def Initialization(config):
    #     if config['seed'] is not None:
    #         torch.manual_seed(config['seed'])  # 检查config['seed']是否为空，不是的话，用torch.manual_seed设定pytorch的种子
    #     # 检查是否有可用的GPU，'cuda'表示选择GPU设备。否则就用CPU
    #     device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    #     # 记录日志，打印当前使用的计算设备
    #     logger.info("Using device: {}".format(device))
    #     # 如果选择cuda设备，获取当前GPU设备的索引编号
    #     if device == 'cuda':
    #         logger.info("Device index: {}".format(torch.cuda.current_device()))
    #     return device
    config['device'] = Initialization(config)
    print_title(f"Threshold Experiments for {config['problem']}")
    logger.info("Loading Data ...")

    if config['problem'] == 'TUAB':
        logger.info("TUAB dataset detected - using memory mapping for data loading...")
        Data = load_tuab_with_memmap(config)
    else:
        Data = Data_Loader(config)

    # print(Data)
    # 6.25修改
    all_results = []
    base_output_dir = config['output_dir']
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_output_dir, f'ThresholdExperiments_{experiment_timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    for threshold in THRESHOLDS:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running experiment with threshold: {threshold * 100}%")
        logger.info(f"{'=' * 80}\n")

        # 为每个threshold创建独立的配置和目录
        threshold_config = config.copy()
        threshold_dir = os.path.join(experiment_dir, f'threshold_{int(threshold * 100)}')
        threshold_config['output_dir'] = threshold_dir
        threshold_config['save_dir'] = os.path.join(threshold_dir, 'checkpoints')
        threshold_config['pred_dir'] = os.path.join(threshold_dir, 'predictions')
        threshold_config['tensorboard_dir'] = os.path.join(threshold_dir, 'tb_summaries')

        # 创建目录
        for dir_path in [threshold_config['save_dir'], threshold_config['pred_dir'],
                         threshold_config['tensorboard_dir']]:
            os.makedirs(dir_path, exist_ok=True)

        # 生成timepoints
        timepoints_dir = os.path.join(threshold_dir, 'important_timepoints')
        os.makedirs(timepoints_dir, exist_ok=True)

        timepoints_path = os.path.join(
            timepoints_dir,
            f"{config['problem']}_threshold_{int(threshold * 100)}_timepoints.pkl"
        )

       # logger.info(f"Generating timepoints with threshold {threshold * 100}%...")
       # generate_important_timepoints_data(
       #     Data, timepoints_path,
       #     target_percentage=0.5,
       #     chunk_count=2,
       #     window_threshold=threshold
       # )
        # 256 samples = 1 second in the experimental pipeline (do not use Table 1 recording rate).
        fs_value = 256
        use_consistency = not args.no_consistency
        if config['problem'] == 'TUAB':
            logger.info(f"TUAB dataset detected - using sampled data for timepoints calculation...")
            sampled_data = {}
            sample_size = 1000
            np.random.seed(42)
            train_indices = np.random.choice(Data['train_data'].shape[0],
                                          min(sample_size, Data['train_data'].shape[0]),
                                          replace=False)

            sampled_data['train_data'] = Data['train_data'][train_indices]
            sampled_data['train_label'] = Data['train_label'][train_indices]

            val_sample_size = min(sample_size, Data['val_data'].shape[0])
            val_indices = np.random.choice(Data['val_data'].shape[0], val_sample_size, replace=False)
            sampled_data['val_data'] = Data['val_data'][val_indices]
            sampled_data['val_label'] = Data['val_label'][val_indices]

            test_sample_size = min(sample_size, Data['test_data'].shape[0])
            test_indices = np.random.choice(Data['test_data'].shape[0], test_sample_size, replace=False)
            sampled_data['test_data'] = Data['test_data'][test_indices]
            sampled_data['test_label'] = Data['test_label'][test_indices]

            all_train_sample_size = min(sample_size, Data['All_train_data'].shape[0])
            all_train_indices = np.random.choice(Data['All_train_data'].shape[0], all_train_sample_size, replace=False)
            sampled_data['All_train_data'] = Data['All_train_data'][all_train_indices]
            sampled_data['All_train_label'] = Data['All_train_label'][all_train_indices]

            logger.info(f"Using {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test, {len(all_train_indices)} all_train samples for timepoints calculation")
            generate_important_timepoints_data(
                sampled_data, timepoints_path,
                target_percentage=0.5,
                chunk_count=2,
                dataset_name=config['problem'],
                window_threshold=threshold,
                fs=fs_value,
                use_consistency=use_consistency,
            )

        else:
            logger.info(f"Generating timepoints with threshold {threshold * 100}%...")
            generate_important_timepoints_data(
                Data, timepoints_path,
                target_percentage=0.5,
                chunk_count=2,
                dataset_name=config['problem'],
                window_threshold=threshold,
                fs=fs_value,
                use_consistency=use_consistency,
            )

        threshold_config['precomputed_timepoints_path'] = timepoints_path

        # 运行实验
        try:
            if threshold_config['Training_mode'] == 'Rep-Learning':
                best_metrics, all_metrics = Rep_Learning(threshold_config, Data)
            elif threshold_config['Training_mode'] == 'Supervised':
                best_metrics, all_metrics = Supervised(threshold_config, Data)

            # 收集结果
            result = {
                'threshold': threshold,
                # Linear Probing结果
                'lp_accuracy': all_metrics.get('linear_probing_acc', 0) * 100,
                'lp_precision': all_metrics.get('linear_probing_precision', 0),
                'lp_recall': all_metrics.get('linear_probing_recall', 0),
                'lp_f1': all_metrics.get('linear_probing_f1', 0),
                'lp_auroc': all_metrics.get('linear_probing_auroc'),
                # Fine-tuning结果
                'ft_accuracy': all_metrics.get('total_accuracy', 0) * 100,
                'ft_precision': all_metrics.get('prec_avg', 0),
                'ft_recall': all_metrics.get('rec_avg', 0),
                'ft_auroc': best_metrics.get('AUROC'),
                'loss': best_metrics.get('loss', 0),
                'output_dir': threshold_dir
            }
            all_results.append(result)

            logger.info(
                f"Threshold {threshold * 100}% completed. LP Acc: {result['lp_accuracy']:.2f}%, FT Acc: {result['ft_accuracy']:.2f}%")

        except Exception as e:
            logger.error(f"Failed to run experiment for threshold {threshold}: {e}")
            import traceback

            traceback.print_exc()

        # ===== 统一保存所有结果 =====
    print_title("Saving All Results")

    # 1. 保存完整的JSON结果
    json_path = os.path.join(experiment_dir, 'all_experiments_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'experiment_info': {
                'dataset': config['problem'],
                'training_mode': config['Training_mode'],
                'timestamp': experiment_timestamp,
                'thresholds': THRESHOLDS,
                'epochs': config['epochs'],
                'batch_size': config['batch_size']
            },
            'results': all_results
        }, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")

    # 2. 创建DataFrame并保存CSV
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('threshold', ascending=False)

    csv_path = os.path.join(experiment_dir, 'threshold_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    logger.info(f"CSV results saved to: {csv_path}")

    # 3. 创建对比表格（分别显示Linear Probing和Fine-tuning）
    print("\n" + "=" * 100)
    print("LINEAR PROBING RESULTS")
    print("=" * 100)
    lp_columns = ['threshold', 'lp_accuracy', 'lp_precision', 'lp_recall', 'lp_f1', 'lp_auroc']
    print(df_results[lp_columns].to_string(index=False))

    print("\n" + "=" * 100)
    print("FINE-TUNING RESULTS")
    print("=" * 100)
    ft_columns = ['threshold', 'ft_accuracy', 'ft_precision', 'ft_recall', 'ft_auroc']
    print(df_results[ft_columns].to_string(index=False))

    # 4. 找出最佳结果
    best_lp_idx = df_results['lp_accuracy'].idxmax()
    best_ft_idx = df_results['ft_accuracy'].idxmax()
    best_lp = df_results.loc[best_lp_idx]
    best_ft = df_results.loc[best_ft_idx]

    print("\n" + "=" * 100)
    print(f"BEST LINEAR PROBING: Threshold {best_lp['threshold'] * 100}% with accuracy {best_lp['lp_accuracy']:.2f}%")
    print(f"BEST FINE-TUNING: Threshold {best_ft['threshold'] * 100}% with accuracy {best_ft['ft_accuracy']:.2f}%")
    print("=" * 100)

    # 5. 创建可视化图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Linear Probing图
    ax1.plot(df_results['threshold'] * 100, df_results['lp_accuracy'], 'b-o',
             markersize=10, linewidth=3, label='Accuracy')

    # 标注最佳点
    ax1.scatter(best_lp['threshold'] * 100, best_lp['lp_accuracy'],
                color='red', s=200, zorder=5)
    ax1.annotate(f'Best: {best_lp["lp_accuracy"]:.2f}%',
                 xy=(best_lp['threshold'] * 100, best_lp['lp_accuracy']),
                 xytext=(10, -10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=11, fontweight='bold')

    ax1.set_xlabel('Window Threshold (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Linear Probing Performance', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([5, 25, 50, 75])

    # Fine-tuning图
    ax2.plot(df_results['threshold'] * 100, df_results['ft_accuracy'], 'r-s',
             markersize=10, linewidth=3, label='Accuracy')

    # 标注最佳点
    ax2.scatter(best_ft['threshold'] * 100, best_ft['ft_accuracy'],
                color='darkgreen', s=200, zorder=5)
    ax2.annotate(f'Best: {best_ft["ft_accuracy"]:.2f}%',
                 xy=(best_ft['threshold'] * 100, best_ft['ft_accuracy']),
                 xytext=(10, -10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                 fontsize=11, fontweight='bold')

    ax2.set_xlabel('Window Threshold (%)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Fine-tuning Performance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([5, 25, 50, 75])

    plt.suptitle(f'Window Threshold Experiments - {config["problem"]} Dataset', fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(experiment_dir, 'threshold_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to: {plot_path}")

    # 6. 生成详细的文本报告
    report_path = os.path.join(experiment_dir, 'experiment_report.txt')
    with open(report_path, 'w') as f:
        f.write("EEG WINDOW THRESHOLD EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {config['problem']}\n")
        f.write(f"Training Mode: {config['Training_mode']}\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(THRESHOLDS)}\n")
        f.write(f"Epochs per experiment: {config['epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("LINEAR PROBING RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(df_results[lp_columns].to_string(index=False))
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("FINE-TUNING RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(df_results[ft_columns].to_string(index=False))
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Best Linear Probing: Threshold {best_lp['threshold'] * 100}% with accuracy {best_lp['lp_accuracy']:.2f}%\n")
        f.write(
            f"Best Fine-tuning: Threshold {best_ft['threshold'] * 100}% with accuracy {best_ft['ft_accuracy']:.2f}%\n")
        f.write(
            f"\nLinear Probing Accuracy Range: {df_results['lp_accuracy'].min():.2f}% - {df_results['lp_accuracy'].max():.2f}%\n")
        f.write(
            f"Fine-tuning Accuracy Range: {df_results['ft_accuracy'].min():.2f}% - {df_results['ft_accuracy'].max():.2f}%\n")

    logger.info(f"Detailed report saved to: {report_path}")

    print_title("All Experiments Completed")
    print(f"\nAll results saved to: {experiment_dir}")

    #if config['Training_mode'] == 'Rep-Learning':
    #    best_aggr_metrics_test, all_metrics = Rep_Learning(config, Data)
    #elif config['Training_mode'] == 'Supervised':
    #    best_aggr_metrics_test, all_metrics = Supervised(config, Data)

    #print_str = 'Best Model Test Summary: '
    #for k, v in best_aggr_metrics_test.items():
    #    print_str += '{}: {} | '.format(k, v)
    #print_title(config['problem'])
    #print(print_str)
    #dic_position_results = [config['problem'], all_metrics['total_accuracy']]
    #All_Results = np.vstack((All_Results, dic_position_results))

#All_Results_df = pd.DataFrame(All_Results)
#All_Results_df.to_csv(os.path.join(config['output_dir'], config['Training_mode'] + '.csv'))




