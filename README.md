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
- `eastmoney_official` P0 公司资料、股本结构和三大财务报表；旧 Eastmoney ann 披露仅保留为未 accepted 对照
- `cninfo_evidence` 通过 8815 版本化接口提供 CNInfo 正式财报证据，不复制 PDF/正文
- `shinny_edb` 信易 EDB 近一年期货分钟线
- `shinny_tqsdk` 信易天勤 TqSdk 只读实时期货与合约元数据



## 这不是主体安装入口

请使用 AI 安装并跑通本项目（通过 [MarketHub](https://github.com/williamxhero/MarketHub) 仓库安装），提示词示例：“阅读 https://github.com/williamxhero/MarketHub/AIREADME.md 并在本机 D:\MarketHub\ 目录中安装这个项目”
