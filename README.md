# ct重建作业

## 运行指南
1.把权重和数据集放到代码目录中，目录结构如下：
```
data
├── TestSet
│   ├── Full_Dose
│   │   ├── xxx.png
│   │   ├── xxx.png
│   │   └── xxx.png
│   └── Low_Dose
│       ├── xxx.png
│       ├── xxx.png
│       └── xxx.png
model
├── xxx.pkl
```
2.打开test.py，填写数据集和权重路径，执行即可复现结果