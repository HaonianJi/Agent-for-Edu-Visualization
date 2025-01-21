## 使用教程

### 零、环境
直接运行
```bash
conda activate pdf-agent
```
如果需要重新安装环境
```bash
conda create -n pdf-agent2 python=3.12
bash install.sh
```

### 一、拆分pdf（适配MMLongBench和LongDocURL，这两个目前已完成）
1. 在config/base.yaml中更改最顶上的数据集名
    ```yaml
    defaults:
        - dataset: <数据集名>
    ```
    
2. 运行
    ```bash
    python scripts/extract.py
    ```

3. 拆分后的图片和文本结果会保存到`tmp/<数据集名>`文件夹中

### 二、用gpt生成新的检索的query（分为 statement-陈述句 和 key_words-关键词）

1. 在config/agent/change_rag_query.yaml中更改system_prompt（已经准备好这两个都prompt）

2. 在config/change_rag_query.yaml中更改question_key，改成'statement'或'key_words'

3. 运行
    ```bash
    python scripts/change_rag_query.py
    ```
4. 结果会保存到`data/<数据集名>/sample-with-retrieval-results.json`中

5. 如果需要换成别的模型，只要在`config/change_rag_query.yaml更改agent.model`就可以了，目前可以选的有openai、qwen2vl、llama31

### 三、进行检索

1. 在config/base.yaml中更改最上面的retrieval类型
    ```yaml
    defaults:
        - retrieval: <类型，text或image>
    ```

2. 可以更改config/base.yaml中的参数
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

### 四、Multiagent推理

1. 设置config/base.yaml中的参数

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

### 五、评估结果（目前是MMLongBench）
1. 设置config/base.yaml中的参数，基本不用更改，只要保证eval_agent.ans_key和前一步时multi_agents.ans_key相同即可
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

## To Do
1. LongDocURL和MP-DocVQA都不是对整个pdf文件进行检索，而是对数据文件中“page_ids“列表中的页面检索，需要在retrieval中增加这一功能

2. MP-DocVQA没有pdf文档，只有拆分后的图片集和ocr文字集，目前存放在tmp中，，需要手工转换格式为`<doc_id>_<page_num>.png`和`<doc_id>_<page_num>.txt`，page_num统一从0开始（MP-DocVQA原始文件名应该也是这样）

3. 检查有没有bug :D