# Kronos_fonny

## 常见问题

遇到错误: 参数未填写

解决方法:

首先需要正确安装 PyTorch 环境,实测可用2025-09-04最新版.

执行:

```
python PyTorch_check.py
```

检查.

然后参考以下步骤配置大脸的下载环境:

```
git clone https://github.com/shiyu-coder/Kronos.git

cd Kronos

#创建虚拟环境,这里取名Kronos(可以修改)
conda create -n Kronos python=3.11
 
conda env list
#激活虚拟环境Kronos
conda activate Kronos

#升级pip
pip install pip -U

#安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#huggingface_hub 库在加载模型时，会尝试使用 safetensors
pip install safetensors -i https://mirrors.aliyun.com/pypi/simple/

#避免模块路径问题, 进入examples下执行.
cd examples/

#国内镜像
export HF_ENDPOINT=https://hf-mirror.com
#执行例子
python  prediction_example.py
```



## MT5客户端的数据需要转换

### csv文件编码需转为 `utf-8-raw`

### 需要补充标题行

```
timestamps,open,high,low,close,volume,amount
```

### 转换时间格式

使用正则替换即可:

```
搜索
^(\d{4})\.(\d{2})\.(\d{2})
替换为
$1-$2-$3
```

