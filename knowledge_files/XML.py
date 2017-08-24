XML最常用：聚合订阅源，博客论坛更新，频繁更新的网站


根元素只能有一个，属性在标签中，字典形式，''
<foo lang='en'>
<bar id='papayawhip' lang="fr">内容</bar>
</foo>

就像Python函数可以在不同的模块(modules)中声明一样，也可以在不同的名字空间(namespace)中声明XML元素。XML文档的名字空间通常看起来像URL。
<feed xmlns='http://www.w3.org/2005/Atom'>

对于XML解析器而言，名字空间相同，元素名相同，属性（或者没有属性）相同，每个元素的文本内容相同，则XML文档相同。

Atom聚合格式被设计成可以包含所有这些信息的标准格式。标题，更新日期，修订

解析：
DOM
SAX
ElementTree