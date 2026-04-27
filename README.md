# MarketHub Packages

这是 MarketHub 的独立 Source Packages 项目。

当前包含 5 个 package：

- `tushare`
- `efinance`
- `mootdx`
- `opentdx`
- `akshare`

每个 package 的源码和 `quotemux_package.json` 放在同一个 Python 子包中，安装后由 QuoteMux 通过 `markethub_packages.<package>` 读取。

未来可以把这个项目放到 GitHub，通过 pip 安装或更新；当前只提供本地项目结构。
