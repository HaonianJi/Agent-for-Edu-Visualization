# LM-MultiAgent-Framework

## 环境
```bash
conda create -n lm-maf python=3.12
conda activate lm-maf
bash install.sh
```

## 教程

### 基础逻辑

主要分为model、agent、multi-agent system三层

- model层负责设置基本的模型，包括模型初始化、模型基本推理方法，可以在`models`目录下添加新的模型，在`config/model`下面设置参数
- agent层负责控制提示词、添加模型的各种推理方法（如self reflect、discuss），可以在`agents/base_agent.py`中添加新的方法，在`config/agent`下面设置提示词
- multi-agent system层负责控制agent的结构关系
    - `predict`方法为单条问题在multi-agent system下的推理流程，需要的参数是question（问题），text（可选，额外的文字信息，是字符串的列表），image（可选，额外的图片信息，是图片的路径的列表）。目前默认的结构是agents列表中的所有agent共同推理，最后sum agent总结回答。若要调整system结构，可以自定义新的predict方法，请参考`agents/custom_method.py`。
    - `predict_dataset`方法是整个数据集在multi-agent system下的推理，但需要根据使用的dataset类来自行定义，可以在`mydatasets`目录下模仿`doc_dataset`自行设置数据集类
    - 其余方法用于调用agent层设置好的各种额外推理方法，请参考注释

### 使用

1. 设置config/base.yaml中的参数

    multi_agents的设置在`config/multi_agents`下，其中`agents`是多个agent、model对组成的列表，可以在对应的config文件夹下面修改agent的prompt和model的参数，sum_agent是最后总结所有agents回答的agent

    如下配置中，agents有两个，一个llama3.1，一个qwen2-VL，汇总的sum_agent是qwen2-VL

    ```yaml
    cuda_visible_devices: '0,1,2,3'

    agents:
        - agent: image_only # agent用来配置prompt，和控制参考资料中文本/图片的使用
        model: qwen2vl # model用来配置使用的模型
        - agent: text_only
        model: llama31
    
    sum_agent:
        agent: sum_agent # 用来汇总所有agent的回答
        model: qwen2vl
    ```

2. 运行
    ```bash
    python scripts/predict_sample.py
    ```

## 进阶教程

### RL训练
1. 在`train`目录下实现`train`的类，在`config/train`下面创建对应的类的`yaml`文件并设置参数，可以参考`grpo`
2. 在`reward`目录下实现`reward`的类，在`config/train`下面创建对应的类的`yaml`文件，可以参考`dummy`
3. 目前实现的`grpo`使用`huggingface`的`trl`库，需要使用的话请安装
    ```bash
    pip install trl
    ```
    并且默认上传`log`至`wandb`，需要使用的话请
    ```bash
    pip install wandb
    wandb login
    ```
4. 目前实现的`grpo`只支持`LLM`，对数据集的要求是必须含有`prompt`参数，用作输入，模型的输出为`completions`参数，可以在`reward`函数中自行使用数据集中的其它参数，该数据集的类请直接使用`datasets`类或者自行在`mydatasets`中添加
5. 运行训练的代码请参考`python scripts/rl_train.py`，参数请参考`config/base_train.yaml`，使用的train类、reward类、用于训练的agent和对应的模型都可以在`config/base_train.yaml`自行修改
6. 运行
    ```bash
    python scripts/rl_train.py
    ```