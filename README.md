# QuoteMux Packages

这是 QuoteMux 的独立 Source Packages 项目。

当前包含 5 个 package：

- `tushare`
- `efinance`
- `mootdx`
- `opentdx`
- `akshare`

每个 package 的源码和 `quotemux_package.json` 放在同一个 Python 子包中，安装后由 QuoteMux 读取。

`tushare` 已内置滑动窗口限速器，默认每分钟最多 `700` 次调用。可以通过环境变量 `MHK_TUSHARE_MAX_CALLS_PER_MINUTE` 调整，设置为 `0` 或负数表示关闭该限速器。

未来可以把这个项目放到 GitHub，通过 pip 安装或更新；当前只提供本地项目结构。
