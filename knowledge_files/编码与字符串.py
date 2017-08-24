源代码用ASCII编码

纯文本在网上的流通要求每段纯文本都要搭载编码信息

unicode为表示任意语言的任意字符设计，4个字节数字表示一个字符，问题在浪费
UTF-8/-16是Unicode编码方式
UTF-8解决既有中文又有ASCII编码数据的文档效率较高
ASCII英语等欧语为一个字节，中日韩为两个字节


Unicode编码的字符序列称为string
0-255之间的数字序列称为bytes
bytes字面值的语法：b''，字面值里的每个字节可以是ASCII字符或者从\x00-\xff编码的十六进制数
bytes.decode(编码方式) bytes解码为string
string.encode(编码方式) string编码为bytes

Unicode到UTF-8：可变长编码。英文1字节汉子3字节生僻4-6字节

编码方式：UTF-8,,gb18030,big5……
xml的编码方式：
HTML的
'''
python2用ASCII,3用UTF-8默认
#‐*‐ coding: windows-1252 ‐*‐
'''
len(bytes) 字节数
len(strings) 字符数

字符串方法
sentence.split('@') #以@分割返回列表，不包括@
字符串分片切片类似列表

原始字符串，无转义r''

string.split(None,3) #NONE空格分隔符，分割3次，即从左数每遇到一个空格分割一次
address.replace("http://", "")去掉前缀

translation_table={ord('a'):ord('o')} {65:79} ASCII
'mark'.translate(translation_table) 'mork'

'''格式化字符串'''
hash = {'name':'hoho','age':18}
'my name is {} ,age {}'.format('hoho',18)
'my name is {1} ,age {0}'.format(10,'hoho')
'my name is {name},age is {age}'.format(name='hoho',age=19)
'my name is {name},age is {age}'.format(**hash)

%s string
%d int
%r

格式说明符{:>4}的意思是“使用最多四个空格使之右对齐
字符串方法rstrip()可以去掉尾随的空白符，包括回车