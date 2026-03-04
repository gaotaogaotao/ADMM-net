## 文件目录结构
d:PATH
├── ADMM.py          # 主入口文件（训练/测试）
├── dataset.py       # 数据加载模块
├── model.py         # U-Net神经网络模型
├── admm.py          # ADMM优化模块
├── utils.py         # 工具函数
├── train.py         # 独立训练脚本
├── test.py          # 独立测试脚本
├── train/           # 训练数据集
│   └── GOPR0372_07_00/
│       ├── blur/    # 模糊图像
│       └── sharp/   # 清晰图像
└── test/            # 测试数据集
    └── GOPR0384_11_00/
        ├── blur/
        └── sharp/

# 输入指令操作
python train.py --root_dir . --model_type admm --epochs 200 --batch_size 8