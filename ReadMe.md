ProjectName     
│ readme 项目说明文档     
│ requirements.txt 存放依赖的外部Python包列表     
│ setup.py 安装、部署、打包的脚本      
├─ bin 存放脚本，执行文件等       
│ └─ projectname        
├─ docs 文档和配置       
│ └─ abc.rst        
│ └─ conf.py 配置文件       
└─ projectname 工程源码（包括源码、测试代码等）     
    │ main.py 程序入口              
    │ init.py       
    └─ tests 测试代码       
        └─ test_main.py     
        └─ init.py      



## 音频特征提取

使用 `sound-net` 提取层 17



## 特征融合

按照融合与预测的先后顺序，分类为早融合(Early fusion)和晚融合(Late fusion)

#### 早融合

先融合多层的特征，然后在融合后的特征上训练预测器（只在完全融合之后，才统一进行检测）。这类方法也被称为skip connection，即采用concat、add操作。这一思路的代表是Inside-Outside Net(ION)和HyperNet。 两个经典的特征融合方法：

（1）concat：系列特征融合，直接将两个特征进行连接。两个输入特征x和y的维数若为p和q，输出特征z的维数为p+q；

（2）add：并行策略，将这两个特征向量组合成复向量，对于输入特征x和y，z = x + iy，其中i是虚数单位。

#### 晚融合

通过结合不同层的检测结果改进检测性能（尚未完成最终的融合之前，在部分融合的层上就开始进行检测，会有多层的检测，最终将多个检测结果进行融合）。这一类研究思路的代表有两种：

（1）feature不融合，多尺度的feture分别进行预测，然后对预测结果进行综合，如Single Shot MultiBox Detector (SSD) , Multi-scale CNN(MS-CNN)

（2）feature进行金字塔融合，融合后进行预测，如Feature Pyramid Network(FPN)等。