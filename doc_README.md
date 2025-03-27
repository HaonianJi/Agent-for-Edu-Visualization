### 零、环境和数据
```bash
conda create -n lm-maf python=3.12
conda activate lm-maf
bash install.sh
bash install_retrieval.sh
mkdir data
cd data
```

从[huggingface](https://huggingface.co/datasets/Lillianwei/Mdocagent-dataset)下载资料，放入`data`文件夹中，其中`PaperText`的`document`文件夹与`papertab`一致，请使用符号链接

### 一、拆分pdf
1. 在config/base_doc.yaml中更改最顶上的数据集名
    ```yaml
    defaults:
        - doc_dataset: <数据集名>
    ```
    
2. 运行
    ```bash
    python scripts/extract.py
    ```

3. 拆分后的图片和文本结果会保存到`tmp/<数据集名>`文件夹中

### 二、进行检索

1. 在config/base_doc.yaml中更改最上面的retrieval类型
    ```yaml
    defaults:
        - retrieval: <类型，text或image>
    ```

2. 可以更改config/base_doc.yaml中的参数
    ```yaml
    retrieval:
    top_k: 10 # 检索时给出top几的结果
    ```

3. 运行
    ```bash
    python scripts/retrieve.py
    ```

4. 检索分为预处理和检索

    text预处理结果在`.ragatouille`，索引路径保存进`sample-with-retrieval-results.json`中，image预处理结果在`tmp/ColpaliRerieval/<image_question_key>`中

    检索结果均保存在`sample-with-retrieval-results.json`中

    注意`MDocAgent`是要分别对`image`和`text`各进行一遍检索

### 三、Multiagent推理

1. 设置config/base_doc.yaml中的参数

    `multi_agents`参数请到对应的`config/multi_agents`文件夹下调整，可以在对应的`config/agent`和`config/model`文件夹下面修改`agent`的`prompt`和`model`的参数，sum_agent是最后总结所有agents回答的agent

2. 运行
    ```bash
    python scripts/predict.py
    ```

3. 结果会保存到`results/<数据集名>/<run-name>/<时间>.json`中

### 四、评估结果
1. 设置config/base_doc.yaml中的参数，基本不用更改，只要保证eval_agent.ans_key和前一步时multi_agents.ans_key相同即可
    ```yaml
    eval_agent: # 用来测试结果
    truncate_len: null # 用来debug，正常使用时设置为null
    ans_key: ans_${run-name}
    agent: base
    model: openai
    ```

2. 运行
    ```bash
    python scripts/eval.py
    ```

3. 结果会保存到`results/<数据集名>/<run-name>/results.txt`中

### 五、其它

RL训练请参考`README`最后，但目前还未支持`vlm`的`grpo`，请自行添加🥹
