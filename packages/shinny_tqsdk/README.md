# shinny_tqsdk

`shinny_tqsdk` 是 QuoteMux 的只读 TqSdk P0 source package。它只使用
`TqApi(auth=TqAuth(username, password))` 的行情与合约查询接口；不创建
交易账户对象、不下单，也不保存任何行情或凭据。

## Capabilities

- `futures.contracts.catalog`: `FUTURE` 合约目录和规格，默认当前未下市，也可由
  core 请求包含已下市合约；直接来自
  `query_quotes(...)/query_symbol_info(...)`。未归一化的原始行保留在
  `raw_metadata`，而序列化所需的 `NaN`/`NaT` 会变为 `null`。
- `futures.contracts.main_mapping`: 当前主力连续对应的真实交割合约。指定品种时
  订阅 `KQ.m@{exchange}.{product}` 并读取 `underlying_symbol`；未指定时使用
  `query_cont_quotes()` 返回的当前真实合约。
- `futures.quotes.main_continuous.realtime`: 主力连续合约实时快照。
- `futures.quotes.contract.realtime`: 显式指定真实交割合约（`EXCHANGE.contract`）
  的完整实时快照，包含可用的五档价格/数量、结算、涨跌停、成交额和交易状态。

多个实时合约一次通过 `get_quote_list()` 在一个 `TqApi` 连接中订阅。每个请求
都会在 `timeout_seconds` 截止前等待所有订阅都有行情时间，并在 `finally` 中关闭
API；没有收到完整行情会报 `TimeoutError`，不会返回看似有效的空快照。

## Configuration and boundaries

配置 `username`（普通配置）和 `password`（secret），可选
`timeout_seconds`（默认 5 秒）。TqSdk 的具体实时行情、合约查询权限取决于天勤账户
的授权；本包不绕过此限制。目录和主力映射的缓存、刷新策略与对外 API 均由 QuoteMux
core 决定，本包不缓存、拼接历史主连或计算任何派生指标。

字段按 TqSdk 原样映射。TqSdk 可能在未收到行情前给出 `NaN` 或 `0`，本包对价格和
数量中的 `NaN` 返回 `null`。中金所（CFFEX）不提供涨跌停价，因此
`upper_limit`/`lower_limit` 可以为 `null`；其他字段也会因交易所或账户权限而缺失。
