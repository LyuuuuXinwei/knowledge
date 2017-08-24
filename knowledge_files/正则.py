正则表达式：字符串查找替换解析文本
import re
re.sub('正则','替换结果',) 正则表被替换目标和其特点格式
re.search()
re.findall('string','正则') # 以列表形式返回全部能匹配正则模式的子串，常用'提取'HTML/URL中的网址/用户信息等,返回（）中的组
re.split(pattern,'string') # 以正则切分并删除正则匹配部分，剩下的变为list
^
$
r'bxxx' 单词边界
\d 数字
x? 0或1个
x* 0或多个
x+ 1或多个
'^M{0,3}$' # 出现0-3次M
(a|b) 或
[ab] ab中一个

() 分组
pattern=re.compile(r'') # 将正则表达式的字符串形式编译，预编译
pattern.search.group() 返回一个元组，元素为分组
用途：检查用户提交信息的格式正确性
