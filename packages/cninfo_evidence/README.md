# cninfo_evidence

`stocks.finance.report_disclosures` 的单一正式 Provider adapter。它只读取 8815 的版本化 `/api/v1/disclosures` JSON，返回强类型 evidence envelope；不下载、复制或缓存 PDF、文章正文或公告原文。

`base_url` 必须通过 QuoteMux runtime profile 显式配置。旧 Eastmoney ann 实现不是 fallback。
