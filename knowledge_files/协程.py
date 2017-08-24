

yield x 返回的x赋给调用生成器next/send的请求者xx=func,send(None/请求)的xx
func.send(s)将s赋值给yield x 这一表达式，如果有ss=yield x，那就是把s赋值给ss

互相传来传去，两个函数协作完成，协程