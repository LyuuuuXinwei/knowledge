with open('plural4‐rules.txt', encoding='utf‐8','r') as pattern_file: open返回一个流对象
不论操作系统，斜杠都正确/，不使用斜杠的是相对路径
磁盘上的文件不是Unicode编码的字符序列，文件是字节序列
默认CP‐1252解码 import locale  locale.getpreferredencoding()

file.name
file.encoding
file.mode
file.read(3) 可选参数：要读取的字符个数
file.tell()
file.seek(0) 定位到特定字节，算是改变Read的定位
mode='w'重写
mode='a'追加
mode='rb'二进制打开图片等文件
文件不存在会自动创建

流对象是一个迭代器，可以for，读行

非文件来源的流对象

只要对象包含read()方法，这个方法使用一个可选参数size并且返回值为一个串，它就是是流对象。
不使size参数调用read()的时候，这个方法应该从输入源读取所有可读的信息然后以单独的一个值返回所有数据
能够“读取”的输入源可以是任何东西：网页，内存中的字符串，甚至是另外一个程序的输出。

stringIO 把内存中的字串当做文件处理
import io io.StringIO('xxxxx')
io.ByteIO 把字节当作二进制文件处理

gzip和bzip2读写压缩文件

with的嵌套
with open() as file ,Redirectstdouto(file):
with open() as file:
    with Redirectstdouto(file):

# Redirectstdouto实现了__enter__,__exit__,分别在with开始和结束时调用

sys.stdin,stdout,stderr
标准输入输出错误，到每一个类UNIX操作系统中的两个管道(pipe)
print()本质：sys.stdout.write()加自动回车
