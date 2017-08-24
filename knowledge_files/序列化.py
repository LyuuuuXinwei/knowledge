再启动返回退出的地方：程序捕获了一个内存中的数据结构并存在磁盘中，下次加载出来

pickle模块的工作对象是数据结构（支持Python原生类型，函数类对象列表元组字典嵌套）

'''保存数据到pickle文件'''
import pickle
with open('entry.pickle','wb') as f: #二进制模式打开
    pickle.dump(entry,f)
dump(可序列化的Python数据结构，将存入的目标文件)

用pickle协议序列化，Python特定，不兼容，序列化为二进制数据

'''读取数据从pickle文件'''
with open('entry.pickle','rb') as f:
    data=f.pickle.load(f)

'''不用文件序列化'''
b=pickle.dumps(obj) #简单返回序列化的数据，bytes对象
obj=pickle.loads(b)

'''另一种兼容的序列化格式：JSON'''
区别：JSON数据格式基于文本而非二进制数据，RFC 4627 定义了JSON格式以及怎样将各种类型的数据编码成文本
求忽略值之间的任意空白
JSON Unicode编码

import json
with open('filename.json',mode='w',encoding='utf-8') as f:
    json.dump(data,f,indent=2)
#JSON文件比pickle更加可读,indent以文档变大为代价使其更可读，0代表每个值单独一行，n代表每个值单独一行并用n个空格缩进嵌套

JSON不支持bytes对象，可以自己定制迷你序列化格式

with open('filename.json',mode='r',encoding='utf-8') as f:
    data=json.load(f)