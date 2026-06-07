# QuoteMux_Packages

`QuoteMux` 的远程 source packages 仓库。

仓库地址：
- [williamxhero/QuoteMux_Packages](https://github.com/williamxhero/QuoteMux_Packages)

## 这是什么

这个仓库只负责维护全部 source packages 的源码与 manifest。

当前 package 以同一 Python 分发包的形式发布，`QuoteMux` 会整体安装或更新这个仓库，然后自动发现其中的 packages。

当前包含：

- `tushare`
- `efinance`
- `mootdx`
- `opentdx`
- `akshare`
- `derived_core`

## 这不是主体安装入口

普通使用者通常不需要直接进入这个仓库执行安装命令。

推荐安装路径是：

1. 先安装 `QuoteMux` 主体，仓库见 [williamxhero/QuoteMux](https://github.com/williamxhero/QuoteMux)
2. 再由 `QuoteMux.install_all_packages()` 或 `MarketHub Admin` 一键安装这里的全部 packages

也就是说，默认入口不是“手工 pip 安装单个 package”，而是由主体仓库统一在线安装整个仓库。

## 一键安装如何发生

有两条标准路径：

- 代码路径：在 `QuoteMux` 中调用 `install_all_packages()`
- GUI 路径：在 `MarketHub` 的 `/admin` 中点击 `安装或更新全部 Packages`

这两条路径最终都会整体安装或更新本仓库。

## 仓库结构

每个 package 目录内都包含：

- `quotemux_package.json`
- `requirements.txt`
- Python 源码

安装后由 `QuoteMux` 自动读取 manifest 并注册能力。

## 与其他仓库的关系

- `QuoteMux`：主体运行时，负责安装和加载本仓库
- `MarketHub`：Admin/GUI 壳，调用 `QuoteMux` 的安装能力
- `QuoteMux_Packages`：提供全部远程 packages 源码
