# QuoteMux_Packages



## 这是什么

这个仓库只负责维护全部 source packages 的源码与 manifest。

当前 package 以同一 Python 分发包的形式发布，`QuoteMux` 会整体安装或更新这个仓库，然后自动发现其中的 packages。

当前包含：

- `tushare`
- `efinance`
- `mootdx`
- `opentdx`
- `akshare`
- `derived_core` 由以上源推导出来的数据
- `crawler_provider` 爬虫数据



## 这不是主体安装入口

请使用 AI 安装并跑通本项目（通过 MarketHub 仓库安装），提示词示例：“阅读 https://github.com/williamxhero/MarketHub/AIREADME.md 并在本机 D:\MarketHub\ 目录中安装这个项目”
