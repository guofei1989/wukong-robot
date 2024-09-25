"""
Author: Guo Fei
Date: 2024-09-13 15:05:18
LastEditors: Guo Fei
LastEditTime: 2024-09-13 15:05:19
Description: 

Copyright (c) 2024 by SDARI, All Rights Reserved. 
"""

import os
from http import HTTPStatus
import json
import re
import logging

import dashscope
from dotenv import load_dotenv

load_dotenv("../..")

from openai import OpenAI
import os

logger = logging.getLogger(__name__)


def get_token():
    api_key = os.getenv("TONGYI_API_KEY")   # 如果您没有配置环境变量,请在此处用您的API Key进行替换

    if api_key is None:
        logger.critical("TONGYI_API_KEY is not set")
    return api_key


def getTongyi(query: str):
    api_key = get_token()
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen2-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": "你是一个语义理解小助手,能从文本信息中提取用户的意图和槽位信息。",
            },
            {
                "role": "user",
                "content": """
                                请从用户输入的文本中提取出两个关键信息槽位:槽位1:动作（Action）- 文本中描述的主要行为或动作。  
                                槽位2（Object）:动作执行的对象- 该动作所作用或针对的具体对象。 
                                返回结果的格式为json字符串, 槽位1和槽位2对应的key分别为action和object,若未提取到,值默认为空字符串。
                                示例1:
                                用户输入:请介绍一下智能机舱模块
                                返回结果: {"action": "介绍", "object": "智能机舱模块"}
                                示例2:
                                用户输入:分析下航速优化的具体功能
                                返回结果:{"action": "分析", "object": "航速优化的具体功能"}
                                现在,请对一下文本内容执行相同的任务,只需要返回json结果。
                                文本内容:
                            """
                + query,
            },
        ],
    )
    try:
        completion = completion.model_dump_json()
        content = json.loads(completion).get("choices")[0].get("message").get("content")
        content = json.loads(re.search(r"{.+}", content).group(0))
        print(content)
        return content
    except Exception as e:
        return {"action": "", "object": ""}






if __name__ == "__main__":
    query = "打开百度"
    print(getTongyi(query))
