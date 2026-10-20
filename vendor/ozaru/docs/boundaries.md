# Who does what

Ozaru is three small, independent utilities. Everything else belongs to the host.

| | Does | Never |
|---|---|---|
| **Porunga** (Ozaru) | turns a JSON identity into a stable key; stores and returns data, files and opaque metadata; checks storage integrity | interprets identities or metadata, knows about software, versions, mocks or who reads what, runs anything |
| **Mimezia** (Ozaru) | fake computations: modes, stubs, the `mock` label, the banner | decides whether a mock is acceptable |
| **Yanus** (Ozaru) | checks that tools, packages and environment variables are there | installs or launches anything |
| **Side labs** (`side_labs/` in a host) | optional, expensive calculations; their own books, chapters, run directories, logs; publish each result with `porunga.save(key(identity), …)` | know which notebook reads a result; define notebook execution |
| **Notebooks** (`notebooks/` in a host) | the analysis, interpretation and presentation — authoritative; describe the result they need, derive its key, `check`, then load, compute directly, mock explicitly, skip, or explain what is missing | import, launch, discover or name a side lab; read run directories |

**The only exchange** between a side lab and a notebook is a Porunga key and its payload. Both sides write the same
plain-JSON identity — every input that changes the result, including software versions and a `mock` flag — in
their own code, from the host's shared design data; neither imports the other. `porunga/docs/identity.md` specifies
the canonical form and the hash.
