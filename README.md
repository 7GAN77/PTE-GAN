PTE-GAN 使用说明
PTE-GAN 用于多类别地质结构的三维重建。本代码以七类沉积地层为例，将中心剖面由厚度 1 逐步拓展至厚度 25。

1. 环境与文件
依赖 Python、NumPy 和 PyTorch。正式实验使用 GPU，请安装与本机环境匹配的 CUDA 版 PyTorch。

将所有 Python 文件和本 README 放在同一目录。以下命令采用简化文件名，请先去掉文件名中的日期和副本编号：

单阶段训练：train_pairs_1to3.py、train_pairs_3to5.py，依次至 train_pairs_23to25.py。
总控训练：run_train_all_stages_1to25.py。
级联生成：将上传的长文件名生成脚本重命名为 generate_cascade_228_7class.py。
2. 正式训练
参考数据放置为：

TEXT
复制
dataset/diceng_228_228_228_zyx_change_xiangsu.npy
数组顺序为 (Z, Y, X)，类别值为 1—7。先检查阶段文件，再启动训练：

BASH
复制
python run_train_all_stages_1to25.py --dry-run
python run_train_all_stages_1to25.py
训练过程生成阶段样本、数据划分文件和模型权重；权重保存在 checkpoints_各阶段_cgan/ 目录。正式数据需另行准备，快速测试生成的数据不能直接替代默认实验数据。

3. 级联生成
生成前需准备全部 12 个阶段的模型权重、上述参考数据，以及：

TEXT
复制
Ti/diceng_228_228_228_zyx_change_xiangsu_insert16_UNKNOWN0.npy
Ti/split_pairs_fixed_test_xy_25_50_75_100_125_150_175_200_seed1234.json
第一个文件为条件体，已知类别为 1—7，未知值为 0，形状须与参考数据一致；第二个文件由训练脚本生成。

为划分文件中的测试剖面各生成 100 组结果：

BASH
复制
python generate_cascade_228_7class.py --device cuda:0 --num_realizations 100
结果保存在 output_mult100/stage12_23to25/。如仅需各生成一组，将 100 改为 1。
