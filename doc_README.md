### 零、环境
```bash
conda create -n lm-maf python=3.12
conda activate lm-maf
bash install.sh
bash install_retrieval.sh
```

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

    # 文字/图片检索时使用的检索键名，可以改成'statement'或'key_words'，前提是已经在第二步生成对应的新的检索query
    text_question_key: question
    image_question_key: question
    ```

3. 运行
    ```bash
    python scripts/retrieve.py
    ```

4. 检索分为预处理和检索

    text预处理结果在`.ragatouille`，索引路径保存进`sample-with-retrieval-results.json`中，image预处理结果在`tmp/ColpaliRerieval/<image_question_key>`中

    检索结果均保存在`sample-with-retrieval-results.json`中

### 三、Multiagent推理

1. 设置config/base_doc.yaml中的参数

    multi_agents.agents下面是多个agent、model对组成的列表，可以在对应的config文件夹下面修改agent的prompt和model的参数，sum_agent是最后总结所有agents回答的agent

    如下配置中，agents有两个，一个llama3.1，一个qwen2-VL，汇总的sum_agent是qwen2-VL

    ```yaml
    multi_agents:
        cuda_visible_devices: '0,1,2,3'
        truncate_len: 1 # 用来debug，正常使用时设置为null
        ans_key: ans_${run-name} # predict时生成的答案的key
        save_message: false # 改成true会再记录所有agent的回答

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
