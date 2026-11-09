# Explicit exchange lifecycle capture

`capture_stock_authority(("SHSE", "SZSE"), limits=CaptureLimits(),
output_dir=None, transport=None)` returns a native `ExchangeAuthorityCapture`.
The scope must be an already sorted, unique, nonempty tuple of those exchanges;
aliases, BJSE and global requests fail before HTTP or artifact creation. The
existing QuoteMux strict public read/provider boundary guards custom transports
as well as the default direct HTTPS transport.

This package does not register `stocks.catalog`, publish a directory or mutate
authority DB tables. No raw Tushare fields, calendar fresh-through or authority
input IDs are invented. Later v2 authority code must validate this capture and
bind the required completed trading day and its own canonical input ID.

The capture has no cache, retries, source fallback, parallel requests, provider
installation or implicit disk output. Each requested market must complete L/P/D
pagination with stable actual metadata and unique identities. All selected first
pages are read again after the partition scans. This proves a bounded paged
observation, not an atomic source snapshot. Limits bound pages (including
verification), response bytes, socket/request time and total monotonic budget.
Status/header/chunk/trailer reads recheck the deadline inside framing loops.
OS DNS resolution runs in a daemon with a bounded caller wait because getaddrinfo
does not honor socket timeouts; an abandoned DNS lookup can finish later but
cannot initiate TCP/HTTPS. Connect attempts and TLS remain on the guarded caller.

`result.to_payload()` is JSON serializable:

- `format_version=exchange-official-stock-authority-capture-v1`;
  `provider=exchange_official`, `contract_version=exchange-official-lifecycle-v1`.
- `scope={exchanges:[...],security_type:A_SHARE,lifecycle:[L,P,D]}` and actual
  capture observation UTC interval.
- `transport_mode` is `direct_official_https` only for the default transport;
  injected transports are `caller_supplied_transport`, bound into the capture
  hash. Offline fixtures must never be accepted as direct authority evidence.
- Each partition has raw/accepted/rejected accounting and page receipts with
  exact URL/parameters, HTTP headers/status, observation interval, actual page
  metadata, schema/selected-page hashes and original `raw_base64` plus SHA256.
  Verification pages have `purpose=verification` and are not counted twice.
- Items contain `market,code,name,list_status,listed_date,delisted_date,
  listing_board,industry,area,owned_fields,evidence_refs`. Industry/area are
  explicitly unowned/empty. Board is owned only from direct partition/bk evidence.
  Listing suspension P is not daily trading suspension or a termination date.
- Only SZSE P/D general security reports may ledger strict six-digit `20xxxx`
  B shares as `out_of_scope_szse_b_share`, under the official March2026 code rule
  pinned in `contracts.py`. Original row/page hashes and rejection policy/counts
  remain bound. Malformed rows, unknown identities and all duplicate securities
  fail the whole attempt. An L A-share tab containing B shares also fails.
- SSE L's explicit status5 membership must match the direct P set. The final item
  appears once as P with both raw references and a counted L-to-P overlap ledger.
- `capture_sha256` hashes the native payload before that hash field is added.

Only explicit `output_dir=Path(...)` creates a unique private attempt directory.
It contains fsynced original `raw/NNNN.raw` bytes and a success/failure manifest.
The disk manifest replaces inline raw_base64 with relative raw_file references;
reconstruct those bytes to verify the native capture_sha256. Its separate
manifest_sha256 binds the disk manifest before that field is added. A failed
attempt is never represented as a complete capture. `ExchangeCaptureError`
preserves its cause and all obtained page receipts; transport failures before a
response have no invented receipt. Oversized custom responses remain attached
to the failure but are not written beyond the configured artifact byte budget.
