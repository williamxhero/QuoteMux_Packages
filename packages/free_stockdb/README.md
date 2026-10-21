# free-stockdb source package

This package is deliberately bounded to `stocks.quotes.daily` with explicit
stock codes and one exact historical date. It calls the local HTTP endpoint:

`GET http://HOST:PORT/?cmd=get&t=日k:CODE:YYYYMMDD`

The response is decoded as MessagePack. The default endpoint is
`127.0.0.1:7899`; QuoteMux source-instance `host` and `port` values override it.
The provider does not import free-stockdb's native `.so`/SDK and does not
perform range or latest-date queries. Volume and amount retain the source units
(shares and yuan).
