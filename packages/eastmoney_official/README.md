# eastmoney_official

本 package 只通过 `query` 暴露 P0 Provider 入口，内部按 company、capital、disclosures、statements 分责。请求必须显式指定 `provider=eastmoney_official`，不提供别名或多源 fallback。

固定网络边界：connect timeout 10 秒、request timeout 60 秒；四项最大响应分别为 1/32/8/32 MiB。capital、disclosures 和 statements 的 page size 均为 500，最大 20 页；capital 的 32 MiB 上限允许完整源列表在本地确定性分页，statements 单请求最多覆盖 160 个季度报告期。达到上限时返回 `contract_error`，不截断。

财务数值按源原值解析为 `Decimal`，`unit_identity=eastmoney_source_original_unscaled`。本 contract 不宣称金额已统一为元或任何币种，也不执行缩放；完整源 row 保留在 `raw_projection`。

QuoteMux 只允许有界短期 cache。股本结构、披露和三表的长期持久化归 STS；本 package 不连接 STS MySQL，MarketHub 不为这些 P0 数据建立 domain/fact 表。
